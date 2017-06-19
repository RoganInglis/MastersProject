import numpy as np
import tensorflow as tf

from models import capacities
from models import BaseModel


class BasicModel(BaseModel):
    def set_model_props(self, config):
        # TODO - Implement this. Should set any model properties specific to this model (i.e. not set in BasicModel)
        print('still to implement set_model_props')

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

            # Load training data (or at least get vocab)
            self.kbp_train_feed_dicts, self.vocab = capacities.load_data(self.placeholders, self.batch_size)  # TODO - edit this to get all data properly

            # Define bicond reader model
            self.logits, self.loss, self.preds = capacities.bicond_reader(self.placeholders, len(self.vocab),
                                                                          self.emb_dim, drop_keep_prob=self.drop_keep_prob)

            # Add train step
            optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # optim = tf.train.AdadeltaOptimizer(learning_rate=1.0)

            if self.l2 != 0.0:
                self.loss = self.loss + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2

            if self.clip is not None:
                gradients = optim.compute_gradients(self.loss)
                if self.clip_op == tf.clip_by_value:
                    capped_gradients = [(tf.clip_by_value(grad, self.clip[0], self.clip[1]), var)
                                        for grad, var in gradients]
                elif self.clip_op == tf.clip_by_norm:
                    capped_gradients = [(tf.clip_by_norm(grad, self.clip), var)
                                        for grad, var in gradients]
                self.train_op = optim.apply_gradients(capped_gradients)
            else:
                self.train_op = optim.minimize(self.loss)

            # Add TensorBoard operations
            self.loss_summary = tf.summary.scalar('Loss', tf.reduce_mean(self.loss, 0))

            self.summary = tf.summary.merge_all()

        return graph

    def infer(self):
        print('still to implement infer')

    def learn_from_epoch(self):
        self.epoch_losses = []

        # Run train_op
        for batch in self.kbp_train_feed_dicts:
            _, current_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=batch)
            self.epoch_losses.append(np.mean(current_loss))

            # Run batch TensorBoard operations here if necessary
            self.summary_writer.add_summary(summary, self.summary_index)
            self.summary_index += 1


