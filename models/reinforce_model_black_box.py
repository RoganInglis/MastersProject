import numpy as np
import tensorflow as tf

from preproc.map import numpify
from models import capacities
from models import BaseModel


class ReinforceModelBlackBox(BaseModel):
    def set_model_props(self, config):
        self.learning_rate_gamma = config['learning_rate_gamma']
        self.lambda_value = config['lambda']
        self.lambda_frac = 1/(1 + self.lambda_value)
        self.alpha = config['alpha']
        self.epsilon = config['epsilon']

        # Initialise parameters
        self.mu = 0  # TODO - depends if batch size > 1 can be used, but random initialisation

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

            self.epoch_loss = tf.placeholder(tf.float32)

            # Load data (or at least get vocab)
            # TODO - This will have to be edited most likely so that full data can be loaded and then processed correctly
            # TODO - switch back to larger data set once testing is complete (remove type argument or switch to train)
            self.kbp_train_data, vocab = capacities.load_data(self.placeholders, 1, data='kbp', type='small_train')
            self.cloze_train_data, self.vocab = capacities.load_data(self.placeholders, 1,
                                                                     vocab=vocab, extend_vocab=True, data='cloze',
                                                                     type='small_train')  # TODO - This is causing memory errors. Could test using less of the cloze data for now but might need to use queues later

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
            self.logits_theta, self.loss_theta, self.preds_theta = capacities.bicond_reader_embedded(self.placeholders,
                                                                                         self.inputs_embedded,
                                                                                         self.emb_dim,
                                                                                         drop_keep_prob=self.drop_keep_prob)

            # Define cloze sampler reader
            with tf.variable_scope('question_reader'):
                self.logits_gamma, self.selection, self.preds_gamma = capacities.question_reader(self.placeholders,
                                                                                                 self.inputs_embedded,
                                                                                                 self.emb_dim,
                                                                                                 drop_keep_prob=self.drop_keep_prob)
            
            # Define cloze sampler reader loss
            targets_index = tf.transpose(
                tf.stack([tf.range(self.batch_size), tf.to_int32(tf.argmax(self.placeholders['targets'], 1))]))
            self.logprob_theta = tf.log(tf.gather_nd(self.preds_theta, targets_index))
            
            # Add theta train step
            optim_theta = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # optim_theta = tf.train.AdadeltaOptimizer(learning_rate=

            if self.l2 != 0.0:
                self.loss_theta = self.loss_theta + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2

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

            # Add gamma update ops
            # Get gamma variables in generator form
            gamma_vars = (var for var in tf.trainable_variables() if 'question_reader' in var.name)

            # Define some variables required for the following loop and for generating the delta noise arrays later
            self.delta_placeholder_shapes = []
            self.delta_placeholders = []
            self.update_gamma = None
            self.add_delta_gamma = None
            self.sub_delta_gamma = None

            # Loop over all gamma variables and add required ops
            for var in gamma_vars:
                # Add summary
                tf.summary.histogram(var.name, var)

                # Get variable shape
                self.delta_placeholder_shapes.append(var.shape)

                # Create placeholder
                delta_placeholder = tf.placeholder('float', shape=var.shape)
                self.delta_placeholders.append(delta_placeholder)

                # Create delta add and subtract ops
                add_op = tf.assign_add(var, delta_placeholder)
                sub_op = tf.assign_sub(var, delta_placeholder)

                # Define gamma update op
                update = self.learning_rate_gamma*(tf.reduce_mean(self.logprob_theta, 0) - self.mu)*delta_placeholder
                new_update_op = tf.assign_sub(var, update)

                tf.summary.histogram(var.name + '_update', update)

                # Group add, subtract and update op
                if self.update_gamma is not None:
                    self.update_gamma = tf.group(new_update_op, self.update_gamma)
                else:
                    self.update_gamma = new_update_op

                if self.add_delta_gamma is not None:
                    self.add_delta_gamma = tf.group(add_op, self.add_delta_gamma)
                else:
                    self.add_delta_gamma = add_op

                if self.sub_delta_gamma is not None:
                    self.sub_delta_gamma = tf.group(sub_op, self.sub_delta_gamma)
                else:
                    self.sub_delta_gamma = sub_op

            # Add TensorBoard operations
            self.loss_theta_summary = tf.summary.scalar('Theta Loss', tf.reduce_mean(self.loss_theta, 0))

            self.summary = tf.summary.merge_all()

        return graph

    def infer(self):
        print('still to implement infer')

    def learn_from_epoch(self):
        self.epoch_losses = []
        done = False

        while not done:  # TODO - done is never true so currently have infinite epoch size
            # Get batch  TODO - edit this to get batches of only cloze or only kbp
            if self.lambda_frac > np.random.uniform():
                # From KBP
                batch, done = self.get_batch(cloze=False)

                # Run only the theta update
                _, current_loss, summary = self.sess.run([self.train_op_theta,
                                                          self.loss_theta, self.summary], feed_dict=batch)
            else:
                # From Cloze
                batch, done = self.get_batch(cloze=True)

                # Run both the theta and gamma updates
                _, _, current_loss, summary = self.sess.run([self.train_op_theta, self.update_gamma,
                                                             self.loss_theta, self.summary], feed_dict=batch)

            self.epoch_losses.append(np.mean(current_loss))

            # Perform baseline update
            logprob = self.sess.run(self.logprob_theta, feed_dict=batch)
            self.mu = self.alpha*self.mu + (1 - self.alpha)*np.mean(logprob)  # TODO - Taking mean of logprob for batch sizes greater than 1, for same batch as used for previous update; double check this is correct

            # Run batch TensorBoard operations here if necessary
            self.summary_writer.add_summary(summary, self.summary_index)
            self.summary_index += 1

        # TODO - should validate/test at the end of each epoch to make sure it's not overfitting

    def get_batch(self, cloze=False):
        """
        Gets a batch of data sampled from both kbp and cloze data according to regularisation parameter lambda.
        If the final data point of either the kbp or cloze data sets is reached while attempting to get a batch
        this will return a smaller batch than batch_size containing the final examples if required
        :return: batch - batch of training data sampled from kbp and cloze data
                 done - boolean flag indicating that the end of the epoch has been reached (i.e. the end of the kbp
                        data has been reached)
        """
        # - Use another reader to decide whether to accept or reject sample while going through shuffled list
        batch = {self.placeholders['answers']: [],
                 self.placeholders['candidates']: [],
                 self.placeholders['question']: [],
                 self.placeholders['question_lengths']: [],
                 self.placeholders['support']: [],
                 self.placeholders['support_lengths']: [],
                 self.placeholders['targets']: []}

        done = False

        # Create noise feed dict
        delta_feed_dict = {}
        for placeholder, size in zip(self.delta_placeholders, self.delta_placeholder_shapes):
            delta_feed_dict[placeholder] = np.random.normal(size=size)
        if cloze:
            # Add noise to gamma
            self.sess.run(self.add_delta_gamma, feed_dict=delta_feed_dict)

        for _ in range(self.batch_size):
            if not cloze:
                # Take single example dict from kbp
                sample = self.kbp_train_data[0]

                # Roll feed dict list
                self.kbp_train_data.append(self.kbp_train_data.pop(0))

                self.epoch_counter += 1
                if self.epoch_counter == len(self.kbp_train_data):
                    done = True
            else:
                while True:
                    # Get candidate from cloze data
                    sample = self.cloze_train_data[0]

                    # Roll feed dict list
                    self.cloze_train_data.append(self.cloze_train_data.pop(0))

                    # Select sample
                    select_predict = self.sess.run(self.preds_gamma, feed_dict={
                        self.placeholders['question']: sample[self.placeholders['question']],
                        self.placeholders['question_lengths']: sample[self.placeholders['question_lengths']]})
                    if np.random.uniform() > select_predict[0][0]:
                        break
            batch = capacities.extend_dict(batch, sample)

        # Make padding even
        capacities.listify_dict(batch)
        batch = numpify(batch)

        # Add the delta feed dict to the batch feed dict
        batch.update(delta_feed_dict)
        if cloze:
            # Subtract noise from gamma
            self.sess.run(self.sub_delta_gamma, feed_dict=delta_feed_dict)

        return batch, done
