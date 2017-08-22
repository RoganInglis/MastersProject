from preproc.vocab import Vocab
from preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
import tensorflow as tf
import numpy as np
import json
import os
import dill as pickle


def dummy_data(sentences=None):
    data = {"question": ["method for task (qa, xxxxx)", "what method is used for qa"], "candidates": [["lstm", "lda", "reasoning"], ["lstm", "lda", "reasoning"]],
            "support": [["in this paper we use an lstm for question answering", "lstm is a good method for qa"],["in this paper we use an lstm for question answering", "lstm is a good method for qa"]],
            "answers": ["lstm", "lstm"]}
    return data


def full_data(file='data\\kbpLocal_train_pos.json'):
    with open(file, encoding='utf8') as data_file:
        data_dict = json.load(data_file)
    data = process_data_dict(data_dict)
    #data = dummy_data()

    return data


def combine_data(list_of_dicts):
    # TODO - This is probably using a lot more memory than it needs to by making a full copy
    if len(list_of_dicts) == 1:
        return list_of_dicts[0]
    else:
        # Combine first 2 elements
        new_list_of_dicts = list_of_dicts[1:]
        for key in list_of_dicts[0]:
            new_list_of_dicts[0][key].extend(list_of_dicts[1][key])

        # Recurse
        return combine_data(new_list_of_dicts)


def process_data_dict(data_dict):
    # TODO - this could probably be sped up
    instances_list = data_dict['instances']
    data_list = []
    for instance in instances_list:
        instance_dict = {**instance['questions'][0], **{'support': instance['support']}}
        for i, answers_instance in enumerate(instance_dict['answers']):
            instance_dict['answers'][i] = answers_instance['text']

        for j, candidates_instance in enumerate(instance_dict['candidates']):
            instance_dict['candidates'][j] = candidates_instance['text']

        for k, support_instance in enumerate(instance_dict['support']):
            instance_dict['support'][k] = support_instance['text']

        data_list.append(instance_dict)

    # Convert list of dicts to dict of lists
    data = {'question': [], 'candidates': [], 'support': [], 'answers': []}
    for dict_instance in data_list:
        data['question'].append(dict_instance['question'])
        data['candidates'].append(dict_instance['candidates'][:])
        data['support'].append(dict_instance['support'])
        data['answers'].append(dict_instance['answers'][0])

    return data


def load_data(placeholders, batch_size=1, vocab=None, extend_vocab=False, file=None, source='kbp', data_type='train',
              data_dir='data\\raw\\'):
    #train_data = dummy_data()
    if file is None:
        list_of_dicts = []
        path = data_dir + source + '\\' + data_type + '\\'
        for file in os.listdir(path):
            if file.endswith('.json'):
                list_of_dicts.append(full_data(path + file))

        data = combine_data(list_of_dicts)
    else:
        data = full_data(file)

    data, vocab = prepare_data(data, vocab=vocab, extend_vocab=extend_vocab)
    #train_feed_dicts = get_feed_dicts(data, placeholders, batch_size=batch_size, bucket_order=None,
    #                                     bucket_structure=None)
    # TODO - get_feed_dicts_old returns a GeneratorWithRestart object containing the data divided in to dicts containing
    # each batch of data. These are padded individually as the padding size only needs to be the same within a batch,
    # however as we want to iterate through and take any example for any batch the padding must be consistent over all
    # the data. get_feed_dicts_old needs to be modified to at least produce batches with consistent padding
    data_np = numpify(data)
    feed_dicts = get_feed_dicts(data_np, placeholders, batch_size, len(data_np['answers']))
    # feed_dicts = get_feed_dicts_old(data, placeholders, batch_size=batch_size, bucket_order=None, bucket_structure=None)
    return feed_dicts, vocab


def load_data_dicts(vocab=None, extend_vocab=False, file=None, source='kbp', data_type='train', data_dir='data\\raw\\'):
    if file is None:
        list_of_dicts = []
        path = data_dir + source + '\\' + data_type + '\\'
        for file in os.listdir(path):
            if file.endswith('.json'):
                list_of_dicts.append(full_data(path + file))

        data = combine_data(list_of_dicts)
    else:
        data = full_data(file)

    data, vocab = prepare_data(data, vocab=vocab, extend_vocab=extend_vocab)

    return data, vocab


def prepare_data(data, vocab=None, extend_vocab=False):
    data_tokenized = deep_map(data, tokenize, ['question', 'support'])
    data_lower = deep_seq_map(data_tokenized, lower, ['question', 'support', 'answers', 'candidates'])
    data = deep_seq_map(data_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["question", "support"])
    if bool(vocab is None) ^ bool(extend_vocab is True):
        if extend_vocab is False:
            vocab = Vocab()

        for instance in data["question"] + data["candidates"] + data["answers"]:
            for token in instance:
                vocab(token)

        for instance in data["support"]:
            for sent in instance:
                for token in sent:
                    vocab(token)
    data = jtr_map_to_targets(data, 'candidates', 'answers')
    data_ids = deep_map(data, vocab, ["question", "candidates", "support", "answers"])
    data_ids = deep_seq_map(data_ids, lambda xs: len(xs), keys=['question', 'support'], fun_name='lengths', expand=True)

    return data_ids, vocab


def listify(data):
    if type(data) is np.ndarray:
        data = data.tolist()
    elif type(data) is list:
        for i, element in enumerate(data):
            data[i] = listify(element)
    return data


def listify_dict(list_dict):
    for key in list_dict:
        listify(list_dict[key])
    return list_dict


def extend_dict(data_dict, new_data_dict):
    for key in data_dict:
        data_dict[key].extend(new_data_dict[key])
    return data_dict


def bicond_reader(placeholders, vocab_size, emb_dim, drop_keep_prob=1.0):
    # [batch_size, max_seq1_length]
    question = placeholders['question']

    # [batch_size, num_sup, max_seq2_length]
    support = placeholders['support']

    # [batch_size, candidate_size]
    targets = tf.to_float(placeholders['targets'])

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("word_embeddings", [vocab_size, emb_dim], dtype=tf.float32)

    with tf.variable_scope("embedders") as varscope:
        question_embedded = tf.nn.embedding_lookup(embeddings, question)
        varscope.reuse_variables()
        support_embedded = tf.nn.embedding_lookup(embeddings, support)
        varscope.reuse_variables()
        candidates_embedded = tf.nn.embedding_lookup(embeddings, candidates)

    dim1s, dim2s, dim3s, dim4s = tf.unstack(
        tf.shape(support_embedded))  # [batch_size, num_supports, max_seq2_length, emb_dim]

    # iterate through all supports
    num_steps = dim2s

    initial_outputs = tf.TensorArray(size=num_steps, dtype='float32')
    initial_i = tf.constant(0, dtype='int32')

    # question_encoding = tf.reduce_sum(question_embedded, 1)

    with tf.variable_scope("conditional_reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(question_embedded, placeholders['question_lengths'], emb_dim,
                                scope=varscope1, drop_keep_prob=drop_keep_prob)

    def should_continue(i, *args):
        # execute the loop for all i supports
        return i < num_steps

    def iteration(i, outputs_):
        # get all instances, take only i-th support, flatten so this becomes a 3-dim tensor
        sup_batchi = tf.reshape(tf.slice(support_embedded, [0, i, 0, 0], [dim1s, 1, dim3s, dim4s]), [dim1s, dim3s, emb_dim])  # [batch_size, num_supports, max_seq2_length, emb_dim]
        sup_lens_batchi = tf.reshape(tf.slice(placeholders['support_lengths'], [0, i], [dim1s, 1]),
                                     [-1])  # [batch_size]

        with tf.variable_scope("conditional_reader_seq2") as varscope2:
            varscope1.reuse_variables()
            # each [batch_size x max_seq_length x output_size]
            outputs, states = reader(sup_batchi, sup_lens_batchi, emb_dim, seq1_states,
                                     scope=varscope2, drop_keep_prob=drop_keep_prob)

        output = tf.concat(axis=1, values=[states[0][1], states[1][1]])

        # squish back into emb_dim num dimensions
        output = tf.contrib.layers.linear(output, emb_dim)

        # batch matrix multiplication to get per-candidate scores
        scores = tf.einsum('bid,bcd->bc', tf.expand_dims(output, 1), candidates_embedded)

        # append scores for the i-th support to previous supports so we can combine scores for all supports later
        outputs_ = outputs_.write(i, scores)

        return i + 1, outputs_

    i, outputs = tf.while_loop(
        should_continue, iteration,
        [initial_i, initial_outputs])

    # packs along axis 0, there doesn't seem to be a way to change that (?)
    outputs_logits = outputs.stack()  # [num_support, batch_size, num_cands]
    scores = tf.reduce_sum(outputs_logits, 0)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=targets)
    predict = tf.nn.softmax(scores)

    return scores, loss, predict


def bicond_reader_embedded(placeholders, inputs_embedded, emb_dim, drop_keep_prob=1.0):
    # [batch_size, candidate_size]
    targets = tf.to_float(placeholders['targets'])

    question_embedded = inputs_embedded['question_embedded']
    support_embedded = inputs_embedded['support_embedded']
    candidates_embedded = inputs_embedded['candidates_embedded']

    dim1s, dim2s, dim3s, dim4s = tf.unstack(
        tf.shape(support_embedded))  # [batch_size, num_supports, max_seq2_length, emb_dim]

    # iterate through all supports
    num_steps = dim2s

    initial_outputs = tf.TensorArray(size=num_steps, dtype='float32')
    initial_i = tf.constant(0, dtype='int32')

    # question_encoding = tf.reduce_sum(question_embedded, 1)

    with tf.variable_scope("conditional_reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(question_embedded, placeholders['question_lengths'], emb_dim,
                                scope=varscope1, drop_keep_prob=drop_keep_prob)

    def should_continue(i, *args):
        # execute the loop for all i supports
        return i < num_steps

    def iteration(i, outputs_):
        # get all instances, take only i-th support, flatten so this becomes a 3-dim tensor
        sup_batchi = tf.reshape(tf.slice(support_embedded, [0, i, 0, 0], [dim1s, 1, dim3s, dim4s]), [dim1s, dim3s, emb_dim])  # [batch_size, num_supports, max_seq2_length, emb_dim]
        sup_lens_batchi = tf.reshape(tf.slice(placeholders['support_lengths'], [0, i], [dim1s, 1]),
                                     [-1])  # [batch_size]

        with tf.variable_scope("conditional_reader_seq2") as varscope2:
            varscope1.reuse_variables()
            # each [batch_size x max_seq_length x output_size]
            outputs, states = reader(sup_batchi, sup_lens_batchi, emb_dim, seq1_states,
                                     scope=varscope2, drop_keep_prob=drop_keep_prob)

        output = tf.concat(axis=1, values=[states[0][1], states[1][1]])

        # squish back into emb_dim num dimensions
        output = tf.contrib.layers.linear(output, emb_dim)

        # batch matrix multiplication to get per-candidate scores
        scores = tf.einsum('bid,bcd->bc', tf.expand_dims(output, 1), candidates_embedded)

        # append scores for the i-th support to previous supports so we can combine scores for all supports later
        outputs_ = outputs_.write(i, scores)

        return i + 1, outputs_

    i, outputs = tf.while_loop(
        should_continue, iteration,
        [initial_i, initial_outputs])

    # packs along axis 0, there doesn't seem to be a way to change that (?)
    outputs_logits = outputs.stack()  # [num_support, batch_size, num_cands]
    scores = tf.reduce_sum(outputs_logits, 0)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=targets)
    predict = tf.nn.softmax(scores)

    return scores, loss, predict


def question_reader(placeholders, inputs_embedded, emb_dim, drop_keep_prob=1.0):
    question_embedded = inputs_embedded['question_embedded']

    with tf.variable_scope('question_reader') as scope:
        outputs, states = reader(question_embedded, placeholders['question_lengths'], emb_dim,
                                 scope=scope, drop_keep_prob=drop_keep_prob)

    # Take final states and concatenate
    output_fw = states[0][1]  # [batch_size, emb_dim]
    output_bw = states[1][1]  # [batch_size, emb_dim]

    output = tf.concat([output_fw, output_bw], 1)  # [batch_size, 2*emb_dim]

    # Linear layer to transform from 2*emb_dim to 2 (select or not select)
    scores = tf.layers.dense(output, 2)  # [batch_size, 2] TODO - need an activation here or just affine transform?

    # Get selection
    selection = tf.argmax(scores, axis=1)  # [batch_size, 1]

    predict = tf.nn.softmax(scores)

    # TODO - what should this return? - definitely selection. loss?, logits?
    return scores, selection, predict


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None, drop_keep_prob=1.0):
    """Dynamic bi-LSTM reader; can be conditioned with initial state of other rnn.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial (forward, backward) states
                                  for the LSTM
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the bi-LSTM.
        States (tensor): The cell states from the bi-LSTM.
    """
    with tf.variable_scope(scope or "reader") as varscope:
        cell_fw = tf.contrib.rnn.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        cell_bw = tf.contrib.rnn.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if drop_keep_prob != 1.0:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=drop_keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=drop_keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # ( (outputs_fw,outputs_bw) , (output_state_fw,output_state_bw) )
        # in case LSTMCell: output_state_fw = (c_fw,h_fw), and output_state_bw = (c_bw,h_bw)
        # each [batch_size x max_seq_length x output_size]
        return outputs, states


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


def write_to_tfrecords_in_chunks(feed_dict_list, base_filename, examples_per_file):
    filename_counter = 0
    for examples in chunker(feed_dict_list, examples_per_file):
        filename = base_filename + str(filename_counter)
        write_to_tfrecords(examples, filename)
        filename_counter += 1


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_sizes_dict(dict_list):
    data_dict = dict_list[0]

    sizes_dict = {}
    for key in data_dict.keys():
        sizes_dict[key.op.name] = data_dict[key][0].shape

    return sizes_dict


def pad_list_of_dicts(list_of_dicts, dict_of_shapes):
    for example_dict in list_of_dicts:
        for key in example_dict.keys():
            # Pad to shape
            current_shape = example_dict[key][0].shape
            shape = dict_of_shapes[key.op.name]
            pad_tuple = tuple(((0, a_shape - current_a_shape) for a_shape, current_a_shape in zip(shape, current_shape)))
            if pad_tuple != ():
                example_dict[key][0] = np.pad(example_dict[key][0], pad_tuple, 'constant')
    return list_of_dicts


def get_max_sizes_dict(sizes_list):
    max_sizes_dict = {}
    for key in sizes_list[0].keys():
        # Create list of sizes for key
        key_sizes = [sizes[key] for sizes in sizes_list]

        # Get max size
        max_sizes_dict[key] = tuple(np.max(np.array(key_sizes), axis=0))
    return max_sizes_dict


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def build_tfrecord_dataset(in_path='data\\raw\\' , out_path='data\\tfrecords\\', examples_per_file=1024):
    """
    Needs to take some pointer to a set of files in json format and load, pre-process and re-save as a set of tfrecord files
    Must also save a file containing the feature shapes and vocab object
    :return:
    """
    # TODO - must load cloze and kbp at same time (so that they share padding size?) and so that vocab can be created and saved correctly
    # TODO - split records files so that no single file is too large? (not sure if necessary or not)
    # Create placeholders
    placeholders = {"question": tf.placeholder(tf.int32, [None, None], name="question"),
                    "question_lengths": tf.placeholder(tf.int32, [None], name="question_lengths"),
                    "candidates": tf.placeholder(tf.int32, [None, None], name="candidates"),
                    "support": tf.placeholder(tf.int32, [None, None, None], name="support"),
                    "support_lengths": tf.placeholder(tf.int32, [None, None], name="support_lengths"),
                    "answers": tf.placeholder(tf.int32, [None], name="answers"),
                    "targets": tf.placeholder(tf.int32, [None, None], name="targets")}

    sizes_list = []

    # Get sizes (this step will take a while)
    for source in ['kbp', 'cloze']:
        for data_type in ['train', 'dev', 'test']:
            # Load
            data, _ = load_data(placeholders, 1, source=source, data_type=data_type,
                                data_dir=in_path)

            # Add Cloze sizes to sizes list
            sizes_list.append(get_sizes_dict(data))

    data = None

    # Get max of sizes
    shapes = get_max_sizes_dict(sizes_list)

    # First load training data and build vocab
    kbp_train_data, vocab = load_data(placeholders, 1, source='kbp', data_type='train', data_dir=in_path)

    # Pad data
    kbp_train_data = pad_list_of_dicts(kbp_train_data, shapes)

    # Save kbp training data as tfrecords
    write_to_tfrecords_in_chunks(kbp_train_data, out_path + 'kbp\\train\\train_kbp', examples_per_file)

    # Add KBP sizes to sizes list
    sizes_list.append(get_sizes_dict(kbp_train_data))

    # Delete KBP training data from memory
    kbp_train_data = None

    cloze_train_data, vocab = load_data(placeholders, 1, vocab=vocab, extend_vocab=True, source='cloze',
                                        data_type='train', data_dir=in_path)  # TODO - appears to not load the whole of the data (loads the same amount as for kbp so maybe reaching a limit for that as well?), maybe try splitting jsons before hand

    # Pad data
    cloze_train_data = pad_list_of_dicts(cloze_train_data, shapes)

    # Save cloze training data as tfrecords
    write_to_tfrecords_in_chunks(cloze_train_data, out_path + 'cloze\\train\\train_cloze', examples_per_file)

    # Add Cloze sizes to sizes list
    sizes_list.append(get_sizes_dict(cloze_train_data))

    # Delete cloze training data from memory
    cloze_train_data = None

    # Load dev and test kbp and cloze using vocab from train
    data_list = ['dev', 'test']
    source_list = ['kbp', 'cloze']
    for data_type in data_list:
        for source in source_list:
            # Load
            data, _ = load_data(placeholders, 1, vocab=vocab, extend_vocab=False, source=source, data_type=data_type,
                                data_dir=in_path)

            # Pad data
            data = pad_list_of_dicts(data, shapes)

            # Save as tfrecords
            write_to_tfrecords_in_chunks(data, out_path + source + '\\' + data_type + '\\' + data_type + '_' + source,
                                         examples_per_file)

    # Delete
    data = None

    # Save shapes and vocab as pkl
    shapes_vocab = {'shapes': shapes, 'vocab': vocab}
    save_obj(shapes_vocab, out_path + 'shapes_vocab')


def create_dict_of_filename_lists(data_dir):
    # Get folder names in data_dir to use as dict keys
    sub_dirs = [name for name in os.listdir(data_dir) if os.path.isdir(data_dir + name)]

    # Get tfrecords filenames
    tfrecords_filenames = [data_dir + file for file in os.listdir(data_dir) if file.endswith('.tfrecords')]

    # Fill dict with lists of filenames (or dicts if more folders)
    if sub_dirs:
        dict_of_filename_lists = {}
        for sub_dir in sub_dirs:
            dict_of_filename_lists[sub_dir] = create_dict_of_filename_lists(data_dir + sub_dir + '/')
    else:
        dict_of_filename_lists = tfrecords_filenames

    return dict_of_filename_lists


def stack_array_lists_in_dict(dict_of_lists):
    for key in dict_of_lists.keys():
        if dict_of_lists[key][0].shape is not ():
            dict_of_lists[key] = np.stack(dict_of_lists[key])
        else:
            dict_of_lists[key] = np.array(dict_of_lists[key])
    return dict_of_lists


if __name__ is '__main__':
    print('test')
