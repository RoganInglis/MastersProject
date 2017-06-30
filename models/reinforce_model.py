import numpy as np
import tensorflow as tf

from models import capacities
from models import BaseModel


class ReinforceModel(BaseModel):
    def set_model_props(self, config):
        self.learning_rate_gamma = config['learning_rate_gamma']
        self.lambda_value = config['lambda']
        self.lambda_frac = 1/(1 + self.lambda_value)
        self.alpha = config['alpha']

        # Initialise parameters
        self.mu = 0  # TODO - depends if batch size > 1 can be used, but random initialisation

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
            self.kbp_train_data, vocab = capacities.load_data(self.placeholders, 1, data='kbp')
            self.cloze_train_data, self.vocab = capacities.load_data(self.placeholders, 1,
                                                                     vocab=vocab, extend_vocab=True, data='cloze')  # TODO - This is causing memory errors. Could test using less of the cloze data for now but might need to use queues later

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
            self.logits_gamma, self.selection, self.preds_gamma = capacities.question_reader(self.placeholders,
                                                                                             self.inputs_embedded,
                                                                                             self.emb_dim,
                                                                                             drop_keep_prob=self.drop_keep_prob)
            
            # Define cloze sampler reader loss
            self.logprob_theta = tf.log(self.preds_theta)

            self.loss_gamma = tf.log(self.preds_gamma)*tf.stop_gradient(self.logprob_theta - self.mu)  # TODO - Double check this is correct
            
            # Add train step
            optim_theta = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # optim_theta = tf.train.AdadeltaOptimizer(learning_rate=

            optim_gamma = tf.train.AdamOptimizer(learning_rate=self.learning_rate_gamma)

            if self.l2 != 0.0:
                self.loss_theta = self.loss_theta + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2
                self.loss_gamma = self.loss_gamma + tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2

            if self.clip is not None:
                gradients_theta = optim_theta.compute_gradients(self.loss_theta)
                gradients_gamma = optim_theta.compute_gradients(self.loss_gamma)
                if self.clip_op == tf.clip_by_value:
                    capped_gradients_theta = [(tf.clip_by_value(grad, self.clip[0], self.clip[1]), var)
                                        for grad, var in gradients_theta]
                    capped_gradients_gamma = [(tf.clip_by_value(grad, self.clip[0], self.clip[1]), var)
                                              for grad, var in gradients_gamma]
                elif self.clip_op == tf.clip_by_norm:
                    capped_gradients_theta = [(tf.clip_by_norm(grad, self.clip), var)
                                        for grad, var in gradients_theta]
                    capped_gradients_gamma = [(tf.clip_by_norm(grad, self.clip), var)
                                              for grad, var in gradients_gamma]
                self.train_op_theta = optim_theta.apply_gradients(capped_gradients_theta)
                self.train_op_gamma = optim_theta.apply_gradients(capped_gradients_gamma)
            else:
                self.train_op_theta = optim_theta.minimize(self.loss_theta)
                self.train_op_gamma = optim_theta.minimize(self.loss_gamma)

            # Add TensorBoard operations
            self.loss_theta_summary = tf.summary.scalar('Theta Loss', tf.reduce_mean(self.loss_theta, 0))
            self.loss_gamma_summary = tf.summary.scalar('Gamma Loss', tf.reduce_mean(self.loss_gamma, 0))

            self.summary = tf.summary.merge_all()

        return graph

    def infer(self):
        print('still to implement infer')

    def learn_from_epoch(self):
        self.epoch_losses = []
        done = False

        while not done:
            # Get batch
            batch, done = self.get_batch()

            # Run train step to update theta and gamma
            _, _, current_loss, summary = self.sess.run([self.train_op_theta, self.train_op_gamma,
                                                         self.loss_theta, self.summary], feed_dict=batch)
            self.epoch_losses.append(np.mean(current_loss))

            # Perform baseline update
            logprob = self.sess.run(self.logprob_theta, feed_dict=batch)
            self.mu = self.alpha*self.mu + (1 - self.alpha)*np.mean(logprob)  # TODO - Taking mean of logprob for batch sizes greater than 1, for same batch as used for previous update; double check this is correct

            # Run batch TensorBoard operations here if necessary
            self.summary_writer.add_summary(summary, self.summary_index)
            self.summary_index += 1

    def get_batch(self):
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

        for _ in range(self.batch_size):
            if self.lambda_frac > np.random.uniform():
                print('kbp')  # TODO - delete
                # Take single example dict from
                try:
                    sample = next(self.kbp_train_data.iterator())  # TODO - deal with the case that the final kbp example is taken (i.e. let learn_from_epoch know to exit loop
                except StopIteration:
                    done = True
                    break

            else:
                print('cloze')  # TODO - delete
                while True:
                    # Get candidate from cloze data
                    try:
                        sample = next(self.cloze_train_data.iterator())  # TODO - deal with case that last example is taken and case that no sample has been selected from whole data (still need to return a sample). Also test that this implementation actually works
                    except StopIteration:
                        done = True
                        break  # TODO - sort out what happens when the end is reached here; would it be better to continuously loop the cloze data regardless of epoch and only end the epoch when the kbp data is used up?

                    # Evaluate using reader
                    select = self.sess.run([self.selection], feed_dict={self.placeholders['question']: sample[self.placeholders['question']],
                                                                        self.placeholders['question_lengths']: sample[self.placeholders['question_lengths']]})  # TODO - CURRENTLY GETTING ISSUE WHERE SELECTION NETWORK MAY BE INITIALISED SUCH THAT IT NEVER SELECTS ANYTHING (maybe need something like decaying epsilon greedy?)

                    # Take example and exit loop or ignore
                    if select == 1:
                        break

            batch = capacities.extend_dict(batch, sample)

        return batch, done
