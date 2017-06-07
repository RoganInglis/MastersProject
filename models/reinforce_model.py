import numpy as np
import tensorflow as tf

from models import capacities
from models import BasicModel


class ReinforceModel(BasicModel):
    def set_model_props(self):
        # TODO - Implement this

    def get_best_config(self):
        # TODO - Implement this

    def get_random_config(fixed_params={}):
        # TODO - Implement this

    def build_graph(self, graph):
        tf.set_random_seed(self.random_seed)

        # Define placeholders
        self.placeholders = {"question": tf.placeholder(tf.int32, [None, None], name="question"),
                             "question_lengths": tf.placeholder(tf.int32, [None], name="question_lengths"),
                             "candidates": tf.placeholder(tf.int32, [None, None], name="candidates"),
                             "support": tf.placeholder(tf.int32, [None, None, None], name="support"),
                             "support_lengths": tf.placeholder(tf.int32, [None, None], name="support_lengths"),
                             "answers": tf.placeholder(tf.int32, [None], name="answers"),
                             "targets": tf.placeholder(tf.int32, [None, None], name="targets")}

        # Define bicond reader model
        """
        need
        train_feed_dicts, vocab, max_epochs=1000, emb_dim=64, l2=0.0, clip=None, clip_op=tf.clip_by_value, sess=None
        """

        logits, loss, preds = capacities.bicond_reader(self.placeholders, len(vocab), emb_dim, drop_keep_prob=self.drop_keep_prob)


        # Add train step
        optim = tf.train.AdamOptimizer(learning_rate=0.001)
        # optim = tf.train.AdadeltaOptimizer(learning_rate=1.0)

        if l2 != 0.0:
            loss = loss + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

        if clip is not None:
            gradients = optim.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                capped_gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                                    for grad, var in gradients]
            elif clip_op == tf.clip_by_norm:
                capped_gradients = [(tf.clip_by_norm(grad, clip), var)
                                    for grad, var in gradients]
            min_op = optim.apply_gradients(capped_gradients)
        else:
            min_op = optim.minimize(loss)

        # Add TensorBoard operations


    def infer(self):

    def learn_from_epoch(self):

    def get_batch(self):
        # TODO - Implement most of the GeneRe specific stuff here?

