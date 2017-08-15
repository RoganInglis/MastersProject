import models.capacities as capacities
import tensorflow as tf
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecords(feed_dict_list, filename):
    # Append filename extension
    filename = filename + '.tfrecords'

    # Define record writer
    writer = tf.python_io.TFRecordWriter(filename)

    # Iterate through feed dict list and write to file
    for example_dict in feed_dict_list:
        # Convert dict entries to features
        feature = {}
        for key in example_dict.keys():
            feat = example_dict[key][0].astype(np.int32)
            feature[key.op.name] = _bytes_feature(tf.compat.as_bytes(feat.tostring()))

        # Create example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialise to string and write to file
        writer.write(example.SerializeToString())

    # Close writer
    writer.close()


def read_from_tfrecords(filename_list, feature_size_dict, num_epochs=1, batch_size=8, capacity=32,
                        num_threads=1, min_after_dequeue=8):
    # Create feature dict to be populated
    feature = {}
    for key in feature_size_dict.keys():
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
    for key in features.keys():
        tensor = tf.decode_raw(features[key], tf.int32)

        # Reshape
        tensor = tf.reshape(tensor, feature_size_dict[key])

        # Append to tensor list
        tensor_list.append(tensor)

    # Create batches by randomly shuffling tensors
    batch = tf.train.shuffle_batch(tensor_list, batch_size=batch_size, capacity=capacity, num_threads=num_threads,
                                   min_after_dequeue=min_after_dequeue)

    return batch


placeholders = {"question": tf.placeholder(tf.int32, [None, None], name="question"),
                "question_lengths": tf.placeholder(tf.int32, [None], name="question_lengths"),
                "candidates": tf.placeholder(tf.int32, [None, None], name="candidates"),
                "support": tf.placeholder(tf.int32, [None, None, None], name="support"),
                "support_lengths": tf.placeholder(tf.int32, [None, None], name="support_lengths"),
                "answers": tf.placeholder(tf.int32, [None], name="answers"),
                "targets": tf.placeholder(tf.int32, [None, None], name="targets")}

kbp_feed_dicts, _ = capacities.load_data(placeholders, 1, data='kbp', type='small_train')

filename = 'data/tfrecords_test'

# Get sizes and types
feature_size_dict = {}
for key in kbp_feed_dicts[0].keys():
    key_name = key.op.name
    size = kbp_feed_dicts[0][key][0].shape
    if size == ():
        feature_size_dict[key_name] = (1,)
    else:
        feature_size_dict[key_name] = size

# Test writing to tfrecords file
write_to_tfrecords(kbp_feed_dicts, filename)

kbp_feed_dicts = None
kbp_train_data = None
vocab = None

# Test reading from tfrecords file to make sure everything is fine
filename_list = [filename + '.tfrecords']

batch = read_from_tfrecords(filename_list, feature_size_dict)

with tf.Session() as sess:
    # Define and run init op
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter('results\\tfrecords_experiments\\', sess.graph)
    summary_writer.add_graph(sess.graph)

    # Create coordinator and queue runner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for _ in range(10):
        test = sess.run(batch)

    print(len(kbp_train_data))
