import os
import sys
import json
import time
import tensorflow as tf
import numpy as np

# See the __init__ script in the models folder
# `make_model` is a helper function to load any models you have
from models import make_model
from tensorflow.contrib.training import HParams
# from hpsearch import hyperband, randomsearch  TODO - For hyperparameter search

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
dir = os.path.dirname(os.path.realpath(__file__))  # TODO - make sure this is right

flags = tf.app.flags


# Hyper-parameters search configuration
flags.DEFINE_boolean('fullsearch', False, 'Perform a full search of hyperparameter space ex:(hyperband -> lr search -> hyperband with best lr)')  # TODO - For hyperparameter search
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')  # TODO - For hyperparameter search
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a HP search')  # TODO - For hyperparameter search

# fixed_params is a trick I use to be able to fix some parameters inside the model random function
# For example, one might want to explore different models fixing the learning rate, see the basic_model get_random_config function
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex: \'{"lr": 0.001}\'')


# Model configuration
flags.DEFINE_string('model_name', 'ReinforceModelBlackBox', 'Unique name of the model')
flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of SGD')
flags.DEFINE_float('drop_keep_prob', 1.0, 'The dropout keep probability')
flags.DEFINE_float('l2', 0.0, 'L2 regularisation strength')
flags.DEFINE_integer('emb_dim', 64, 'Word embedding dimension')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('clip_op', 'value', 'Set type of gradient clipping to use - can be either value or norm')

# GeneRe specific parameters
flags.DEFINE_float('learning_rate_gamma', 1e-3, 'The learning rate of SGD for the sampler parameter update')
flags.DEFINE_float('lambda', 0.1, 'Generative regularisation strength')
flags.DEFINE_float('alpha', 1e-1, 'The baseline update parameter')
flags.DEFINE_float('epsilon', 1e-1, 'The initial exploration parameter value')

# Training configuration
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 1000, 'Number of epochs')
flags.DEFINE_boolean('infer', False, 'Load a model for inference')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + str(int(time.time())), 'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
flags.DEFINE_integer('random_seed', np.random.randint(0, 2**8), 'Value of random seed')


def main(_):
    config = flags.FLAGS.__flags.copy()
    # fixed_params must be a string to be passed in the shell, let's use JSON
    config["fixed_params"] = json.loads(config["fixed_params"])
    config['clip'] = None  # Defined here as it may need to be a list

    if config['fullsearch']:
        # Some code for HP search ... TODO - For hyperparameter search
        print('Hyperparameter search not implemented yet')
    else:
        model = make_model(config)

        if config['infer']:
            # Some code for inference ...
            model.infer()  # TODO - Add inputs if required
        else:
            # Some code for training ...
            model.train()  # TODO - Add inputs if required


if __name__ == '__main__':
    tf.app.run()

