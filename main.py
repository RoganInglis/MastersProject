import os
import json
import time
import tensorflow as tf
import numpy as np

# See the __init__ script in the models folder
# `make_model` is a helper function to load any models you have
from models import make_model

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags


# Hyper-parameters search configuration
flags.DEFINE_boolean('fullsearch', False, 'Perform a full search of hyperparameter space ex:(hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a HP search')

# fixed_params is a trick I use to be able to fix some parameters inside the model random function
# For example, one might want to explore different models fixing the learning rate, see the basic_model get_random_config function
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex: \'{"lr": 0.001}\'')


# Model configuration
flags.DEFINE_string('model_name', 'ReinforceModelBlackBoxReader', 'Unique name of the model')
flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of SGD')
flags.DEFINE_float('drop_keep_prob', 1.0, 'The dropout keep probability')
flags.DEFINE_float('l2', 0.0, 'L2 regularisation strength')
flags.DEFINE_integer('emb_dim', 64, 'Word embedding dimension')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('clip_op', 'value', 'Set type of gradient clipping to use - can be either value or norm')

# GeneRe specific parameters
flags.DEFINE_float('learning_rate_gamma', 1e-2, 'The learning rate of SGD for the sampler parameter update')
flags.DEFINE_float('lambda', 0.5, 'Generative regularisation strength')
flags.DEFINE_float('alpha', 1e-3, 'The baseline update parameter')

# Training configuration
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 1000, 'Number of epochs')
flags.DEFINE_boolean('infer', False, 'Load a model for inference')
flags.DEFINE_boolean('test', False, 'Load a model and compute test performance')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
flags.DEFINE_integer('num_threads', 1, 'Number of threads to be used for each input pipeline')
flags.DEFINE_string('data_dir', 'data/tfrecords/', 'Top level directory for the tfrecords data')
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + str(int(time.time())), 'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
flags.DEFINE_string('dev_result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + str(int(time.time())) + '/dev', 'Name of the directory to store/log the model dev results (if it exists, the model will be loaded from it)')
flags.DEFINE_integer('dev_summary_batch_size', 128, 'Batch size to use when computing dev accuracy')
flags.DEFINE_integer('dev_summary_interval', 10, 'Interval at which to compute dev accuracy and update summary during training')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
flags.DEFINE_integer('random_seed', np.random.randint(0, 2**8), 'Value of random seed')


def main(_):
    config = flags.FLAGS.__flags.copy()
    # fixed_params must be a string to be passed in the shell, let's use JSON
    config["fixed_params"] = json.loads(config["fixed_params"])
    config['clip'] = None  # Defined here as it may need to be a list

    if config['fullsearch']:
        print('Hyperparameter search not implemented yet')
    else:
        model = make_model(config)

        if config['infer']:
            # Some code for inference ...
            model.infer()
        elif config['test']:
            model.test()
        else:
            # Some code for training ...
            model.train()


if __name__ == '__main__':
    tf.app.run()

