import numpy as np
import tensorflow as tf

from preproc.map import numpify
from models import capacities
from models import BaseModel


class ReinforceModelBlackBox(BaseModel):
    def set_model_props(self, config):
        self.learning_rate_gamma = config['learning_rate_gamma']
        self.lambda_value = config['lambda']
        self.lambda_frac = 1 / (1 + self.lambda_value)
        self.alpha = config['alpha']

        # Initialise parameters
        len_gamma = 3
        self.gamma = np.ones(len_gamma)
        self.mu = 0.

        self.delta = np.zeros(len_gamma)

        self.gamma_noise = self.gamma
        self.gamma_noise_distribution = self.gamma

        self.epoch_counter = 0

    def get_best_config(self):
        # TODO - Implement this. Look at example online
        print('still to implement get_best_config')

    def get_random_config(fixed_params={}):
        # TODO - Implement this. Look at example online
        print('still to implement get_random_config')

    def build_graph(self, graph):
        """
        Defines the TensorFlow computation graph to be used later for training and inference
        :param graph: TensorFlow graph e.g. tf.Graph()
        :return: built graph
        """
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            # Define placeholders
            self.placeholders = {"question": tf.placeholder(tf.int32, [None, None], name="question"),
                                 "question_lengths": tf.placeholder(tf.int32, [None], name="question_lengths"),
                                 "candidates": tf.placeholder(tf.int32, [None, None], name="candidates"),
                                 "support": tf.placeholder(tf.int32, [None, None, None], name="support"),
                                 "support_lengths": tf.placeholder(tf.int32, [None, None], name="support_lengths"),
                                 "answers": tf.placeholder(tf.int32, [None], name="answers"),
                                 "targets": tf.placeholder(tf.int32, [None, None], name="targets")}

            self.placeholder_keys = list([key for key in self.placeholders.keys()])

            self.epoch_loss = tf.placeholder(tf.float32)

            # Create additional placeholder for gamma
            self.gamma_placeholder = tf.placeholder('float', [None])
            tf.summary.histogram('Gamma', self.gamma_placeholder)

            # Create input pipelines
            self.kbp_batch = dict()
            self.cloze_batch = dict()
            self.kbp_batch['train'] = self.input_pipeline(self.data_filenames['kbp']['train'])

            # One for each
            self.cloze_batch['train'] = []
            self.cloze_batch['train'].append(self.input_pipeline(self.data_filenames['cloze']['train']['group_0']))
            self.cloze_batch['train'].append(self.input_pipeline(self.data_filenames['cloze']['train']['group_1']))
            self.cloze_batch['train'].append(self.input_pipeline(self.data_filenames['cloze']['train']['group_2']))

            self.kbp_batch['dev'] = self.input_pipeline(self.data_filenames['kbp']['dev'])
            self.cloze_batch['dev'] = self.input_pipeline(self.data_filenames['cloze']['dev'])

            self.kbp_batch['test'] = self.input_pipeline(self.data_filenames['kbp']['test'], num_epochs=1)
            self.cloze_batch['test'] = self.input_pipeline(self.data_filenames['cloze']['test'], num_epochs=1)

            # Embed inputs
            with tf.variable_scope("embeddings"):
                embeddings = tf.get_variable("word_embeddings", [len(self.vocab), self.emb_dim], dtype=tf.float32)

            with tf.variable_scope("embedders") as varscope:
                question_embedded = tf.nn.embedding_lookup(embeddings, self.placeholders['question'])
                varscope.reuse_variables()
                support_embedded = tf.nn.embedding_lookup(embeddings, self.placeholders['support'])
                varscope.reuse_variables()
                candidates_embedded = tf.nn.embedding_lookup(embeddings, self.placeholders['candidates'])

            self.inputs_embedded = {'question_embedded': question_embedded,
                                    'support_embedded': support_embedded,
                                    'candidates_embedded': candidates_embedded}

            # Define bicond reader model
            with tf.variable_scope('bicond_reader'):
                self.logits_theta, self.loss_theta, self.preds_theta = capacities.bicond_reader_embedded(
                    self.placeholders,
                    self.inputs_embedded,
                    self.emb_dim,
                    drop_keep_prob=self.drop_keep_prob)

            prediction = tf.argmax(self.preds_theta, 1)
            targets = tf.argmax(self.placeholders['targets'], 1)
            self.correct = tf.cast(tf.equal(prediction, targets), tf.float32)
            self.accuracy = tf.reduce_mean(self.correct)

            self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

            # Define cloze sampler reader loss
            targets_index = tf.transpose(
                tf.stack([tf.range(self.batch_size), tf.to_int32(tf.argmax(self.placeholders['targets'], 1))]))
            self.logprob_theta = tf.log(tf.gather_nd(self.preds_theta, targets_index))

            # Add theta train step
            optim_theta = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # optim_theta = tf.train.AdadeltaOptimizer(learning_rate=

            if self.l2 != 0.0:
                self.loss_theta = self.loss_theta + tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2

            if self.clip is not None:
                gradients_theta = optim_theta.compute_gradients(self.loss_theta)
                if self.clip_op == tf.clip_by_value:
                    capped_gradients_theta = [(tf.clip_by_value(grad, self.clip[0], self.clip[1]), var)
                                              for grad, var in gradients_theta]
                elif self.clip_op == tf.clip_by_norm:
                    capped_gradients_theta = [(tf.clip_by_norm(grad, self.clip), var)
                                              for grad, var in gradients_theta]
                self.train_op_theta = optim_theta.apply_gradients(capped_gradients_theta)
            else:
                self.train_op_theta = optim_theta.minimize(self.loss_theta)

            # Add TensorBoard operations
            self.loss_theta_summary = tf.summary.scalar('Theta Loss', tf.reduce_mean(self.loss_theta, 0))

            self.summary = tf.summary.merge_all()
            self.dev_summary = tf.summary.merge([self.loss_theta_summary, self.accuracy_summary])

        return graph

    def test(self):
        correct_list = []
        done = False
        while not done:
            # Get test batch
            batch, done = self.get_batch(data_type='test')

            # Compute batch accuracy and correct
            correct = self.sess.run(self.correct, feed_dict=batch)

            correct_list.append(correct)

        total_correct = np.concatenate(correct_list, 0)
        total_accuracy = np.mean(total_correct)

        return total_correct, total_accuracy

    def infer(self):
        print('still to implement infer')

    def learn_from_epoch(self):
        self.epoch_losses = []
        done = False
        best_dev_loss = 10

        while not done:
            if self.summary_index % self.dev_summary_interval == 0 or self.summary_index == 0:
                dev_loss, _ = self.compute_loss_accuracy(self.dev_summary_batch_size)
                if dev_loss < best_dev_loss:
                    self.save()
                    best_dev_loss = dev_loss

            # Get batch
            if self.lambda_frac > np.random.uniform():
                # From KBP
                batch, done = self.get_batch(cloze=False)

                # Run only the theta update
                _, current_loss, summary = self.sess.run([self.train_op_theta,
                                                          self.loss_theta, self.summary], feed_dict=batch)

                logprob = self.sess.run(self.logprob_theta, feed_dict=batch)
            else:
                # Add noise to gamma
                self.delta = np.random.normal(size=len(self.gamma))

                self.gamma_noise = self.gamma + self.delta

                self.gamma_noise_distribution = capacities.vector_to_distribution(self.gamma_noise)

                # From Cloze
                batch, done = self.get_batch(cloze=True)

                # Run both the theta and gamma updates
                _, current_loss, summary = self.sess.run([self.train_op_theta,
                                                          self.loss_theta, self.summary], feed_dict=batch)

                # Update gamma
                logprob = self.sess.run(self.logprob_theta, feed_dict=batch)
                gamma_update = self.learning_rate_gamma * (np.mean(logprob) - self.mu) * self.delta
                self.gamma -= gamma_update

            self.epoch_losses.append(np.mean(current_loss))

            # Perform baseline update
            self.mu = self.alpha * self.mu + (1 - self.alpha) * np.mean(logprob)

            # Run batch TensorBoard operations here if necessary
            self.summary_writer.add_summary(summary, self.summary_index)
            self.summary_index += 1

    def get_batch(self, cloze=False, data_type='train', batch_size=None):
        """
        Gets a batch of data sampled from both kbp and cloze data according to regularisation parameter lambda.
        If the final data point of either the kbp or cloze data sets is reached while attempting to get a batch
        this will return a smaller batch than batch_size containing the final examples if required
        :return: batch - batch of training data sampled from kbp and cloze data
                 done - boolean flag indicating that the end of the epoch has been reached (i.e. the end of the kbp
                        data has been reached)
        """
        if batch_size is None:
            batch_size = self.batch_size

        # - Use another reader to decide whether to accept or reject sample while going through shuffled list
        batch = {self.placeholders['answers']: [],
                 self.placeholders['candidates']: [],
                 self.placeholders['question']: [],
                 self.placeholders['question_lengths']: [],
                 self.placeholders['support']: [],
                 self.placeholders['support_lengths']: [],
                 self.placeholders['targets']: []}

        done = False

        for _ in range(batch_size):
            try:
                if not cloze:
                    # Take single example dict from kbp
                    sample_ops_list = self.batch_dict_to_list(self.kbp_batch[data_type])
                    sample_list = self.sess.run(sample_ops_list)
                    sample = self.batch_list_to_feed_dict(sample_list)

                else:
                    if data_type is 'train':
                        # Select group to take sample from according to gamma vector
                        # Turn gamma into a multinomial distribution
                        cloze_group = np.random.choice(int(len(self.gamma)), p=self.gamma_noise_distribution)

                        # Get sample from correct cloze data group
                        sample_ops_list = self.batch_dict_to_list(self.cloze_batch[data_type][cloze_group])
                        sample_list = self.sess.run(sample_ops_list)
                        sample = self.batch_list_to_feed_dict(sample_list)
                    else:
                        # Get sample from correct cloze data group
                        sample_ops_list = self.batch_dict_to_list(self.cloze_batch[data_type])
                        sample_list = self.sess.run(sample_ops_list)
                        sample = self.batch_list_to_feed_dict(sample_list)

                batch = capacities.extend_dict(batch, sample)
            except Exception as ex:
                self.coord.request_stop(ex)
                done = True

        # Concatenate batch
        batch = capacities.stack_array_lists_in_dict(batch)

        batch[self.gamma_placeholder] = self.gamma

        return batch, done

    def batch_dict_to_list(self, batch_dict):
        batch_list = []
        for key in self.placeholder_keys:
            batch_list.append(batch_dict[key])
        return batch_list

    def batch_list_to_feed_dict(self, batch_list):
        feed_dict = dict()
        for key, item in zip(self.placeholder_keys, batch_list):
            feed_dict[self.placeholders[key]] = item
        return feed_dict

    def compute_loss_accuracy(self, accuracy_batch_size):
        # Get dev batch
        dev_batch, _ = self.get_batch(data_type='dev', batch_size=accuracy_batch_size)

        # Run accuracy and summary ops
        dev_summary, dev_loss, dev_accuracy = self.sess.run([self.dev_summary, self.loss_theta, self.accuracy],
                                                            feed_dict=dev_batch)

        # Write summary
        self.dev_summary_writer.add_summary(dev_summary, self.summary_index)

        return np.mean(dev_loss), dev_accuracy

