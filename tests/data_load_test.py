import tensorflow as tf
import json
from models import capacities

maindir = 'C:\\Users\\rogan\\OneDrive\\Documents\\Machine Learning\\Project\\Rogan-Project\\'
split_save_location = maindir + 'data\\split_data\\'

"""
TRYING LOADING JUST JSONS
"""
"""
filenames = [maindir + 'data\\kbpLocal_train_pos.json',
             maindir + 'data\\kbpLocal_train_neg.json',
             maindir + 'data\\clozeLocal_train_pos.json',
             maindir + 'data\\clozeLocal_train_neg.json',
             maindir + 'data\\clozeLocal_train-002.json']
"""

"""
filenames = [maindir + 'data\\kbpLocal_train_pos.json',
             maindir + 'data\\kbpLocal_train_neg.json',
             maindir + 'data\\clozeLocal_train_pos.json',
             maindir + 'data\\clozeLocal_train_neg.json']
"""

filenames = [maindir + 'data\\clozeLocal_train-002.json']

data_dict_list = []

for file in filenames:
    with open(file, encoding='utf8') as data_file:
        data_dict_list.append(json.load(data_file))

# Split json and save
split_size = 10000


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

data_chunks = chunker(data_dict_list[0]['instances'], split_size)

for i, data_chunk in enumerate(data_chunks):
    new_dict = {'instances': data_chunk}
    with open(split_save_location + 'clozeLocal_train_' + str(i) + '.json', 'w') as outfile:
        json.dump(new_dict, outfile)


"""
TRYING LOADING DATA FULLY
"""
"""
# Create placeholders
placeholders = {"question": tf.placeholder(tf.int32, [None, None], name="question"),
                "question_lengths": tf.placeholder(tf.int32, [None], name="question_lengths"),
                "candidates": tf.placeholder(tf.int32, [None, None], name="candidates"),
                "support": tf.placeholder(tf.int32, [None, None, None], name="support"),
                "support_lengths": tf.placeholder(tf.int32, [None, None], name="support_lengths"),
                "answers": tf.placeholder(tf.int32, [None], name="answers"),
                "targets": tf.placeholder(tf.int32, [None, None], name="targets")}

feed_dicts, vocab = capacities.full_data(placeholders, 1, data='cloze')
"""
