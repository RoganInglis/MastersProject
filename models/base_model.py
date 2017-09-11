import os
import copy
import json
import tensorflow as tf
import numpy as np
from models import capacities


class BaseModel(object):
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        # I like to keep the best HP found so far inside the model itself
        # This is a mechanism to load the best HP and override the configuration
        if config['best']:
            config.update(self.get_best_config())

        # I make a `deepcopy` of the configuration before using it
        # to avoid any potential mutation when I iterate asynchronously over configurations
        self.config = copy.deepcopy(config)

        if config['debug']:  # This is a personal check i like to do
            print('config', self.config)

        # When working with NN, one usually initialize randomly
        # and you want to be able to reproduce your initialization so make sure
        # you store the random seed and actually use it in your TF graph (tf.set_random_seed() for example)
        self.random_seed = self.config['random_seed']

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.num_threads = self.config['num_threads']
        self.result_dir = self.config['result_dir']
        self.dev_result_dir = self.config['dev_result_dir']
        self.dev_summary_batch_size = self.config['dev_summary_batch_size']
        self.dev_summary_interval = self.config['dev_summary_interval']
        self.data_dir = self.config['data_dir']
        self.max_iter = self.config['max_iter']
        self.drop_keep_prob = self.config['drop_keep_prob']
        self.learning_rate = self.config['learning_rate']
        self.l2 = self.config['l2']
        self.emb_dim = self.config['emb_dim']
        self.batch_size = self.config['batch_size']
        self.clip = config['clip']
        if self.config['clip_op'] == 'value':
            self.clip_op = tf.clip_by_value
        elif self.config['clip_op'] == 'norm':
            self.clip_op = tf.clip_by_norm
        else:
            raise Exception('clip_op must be value or norm')

        self.placeholder_keys = []


        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_model_props(config)

        # Load vocab and shapes
        shapes_vocab = capacities.load_obj(self.data_dir + 'shapes_vocab')
        self.shapes = shapes_vocab['shapes']
        self.vocab = shapes_vocab['vocab']

        # Create dict of filename lists for tfrecords input pipeline
        self.data_filenames = capacities.create_dict_of_filename_lists(self.data_dir)

        # Again, child Model should provide its own build_graph function
        self.graph = self.build_graph(tf.Graph())

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=50)
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.summary_writer = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.dev_summary_writer = tf.summary.FileWriter(self.dev_result_dir, self.sess.graph)
        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()

    def set_model_props(self, config):
        # This function is here to be overriden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def get_best_config(self):
        # This function is here to be overriden completely.
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the agent')

    def infer(self):
        raise Exception('The infer function must be overriden by the agent')

    def test(self):
        raise Exception('The test function must be overriden by the agent')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def train(self, save_every=1):
        # This function is usually common to all your models
        self.summary_index = 0
        for self.epoch_id in range(0, self.max_iter):
            self.learn_from_epoch()

            # Perform epoch TensorBoard operations here if necessary

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and self.epoch_id % save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        if self.config['debug']:
            print('Saving to %s' % self.result_dir)
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(self.epoch_id))

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.sess.run(self.init_op)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def input_pipeline(self, filename_list, batch_size=1, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.max_iter

        # Remove non-string entries in filename_list
        filename_list = [x for x in filename_list if type(x) is str]

        # Create feature dict to be populated
        feature = dict()
        for key in self.shapes.keys():
            feature[key] = tf.FixedLenFeature([], tf.string)

        # Create queue from filename list
        filename_queue = tf.train.string_input_producer(filename_list, num_epochs=num_epochs)

        # Define reader
        reader = tf.TFRecordReader()

        # Read next record
        _, serialized_example = reader.read(filename_queue)

        # Decode record
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert strings back to numbers
        tensor_list = []
        for key in self.placeholder_keys:
            tensor = tf.decode_raw(features[key], tf.int32)

            # Reshape
            tensor = tf.reshape(tensor, self.shapes[key])

            # Append to tensor list
            tensor_list.append(tensor)

        # Create batches by randomly shuffling tensors
        batch = tf.train.shuffle_batch(tensor_list, batch_size=batch_size, capacity=32, num_threads=self.num_threads,
                                       min_after_dequeue=8)
        batch_dict = dict()
        for i, key in enumerate(self.placeholder_keys):
            batch_dict[key] = batch[i]

        return batch_dict

