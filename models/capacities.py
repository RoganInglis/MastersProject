from preproc.vocab import Vocab
from preproc.batch import get_feed_dicts
from preproc.map import numpify, tokenize, lower, deep_map, deep_seq_map, jtr_map_to_targets
import tensorflow as tf
import numpy as np
import json
import os
import dill as pickle
import Levenshtein as lsn


def dummy_data():
    """
    :return: A dict containing a single example of data in the correct format, for testing purposes
    """
    data = {"question": ["method for task (qa, xxxxx)", "what method is used for qa"], "candidates": [["lstm", "lda", "reasoning"], ["lstm", "lda", "reasoning"]],
            "support": [["in this paper we use an lstm for question answering", "lstm is a good method for qa"],["in this paper we use an lstm for question answering", "lstm is a good method for qa"]],
            "answers": ["lstm", "lstm"]}
    return data


def full_data(file='data\\kbpLocal_train_pos.json'):
    """
    load data from specified .json file and return in correct dict format
    :param file: path to a .json file containing the correct data
    :return: data - dict with correct keys containing the loaded data (still in string form)
    """
    with open(file, encoding='utf8') as data_file:
        data_dict = json.load(data_file)
    data = process_data_dict(data_dict)
    return data


def combine_data(list_of_dicts):
    """
    Convert a list of dicts into a single dict
    :param list_of_dicts: list of dicts to combine
    :return: dict containing all data combined for each key
    """
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
    """
    Converts the dict format obtained by loading the raw .json data files into the correct format
    :param data_dict: Dict containing data just loaded from .json format
    :return: data - dict containing the data in the correct format
    """
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
    """
    Loads the specified data, preprocesses it and returns the vocab and feed dicts for that data
    :param placeholders: Dict of tensorflow placeholders
    :param batch_size: batch size for returned feed dicts
    :param vocab: existing vocab to use or extend if required
    :param extend_vocab: if passing an existing vocab, add new tokens to the vocab rather than treating them as OOV
    :param file: path to a specific file to load if required
    :param source: source of the data to load. Should be either kbp or cloze
    :param data_type: type of data to load; train, dev or test
    :param data_dir: parent directory of the data (assumed it contains sub folders for source and type e.g. raw\\kbp\\train)
    :return: feed dicts and vocab of the loaded data
    """
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
    data_np = numpify(data)
    feed_dicts = get_feed_dicts(data_np, placeholders, batch_size, len(data_np['answers']))
    return feed_dicts, vocab


def load_data_dicts(vocab=None, extend_vocab=False, file=None, source='kbp', data_type='train', data_dir='data\\raw\\'):
    """
    Performs the same task as load_data but stops before the data is divided into feed dicts
    :param vocab: existing vocab to use or extend if required
    :param extend_vocab: if passing an existing vocab, add new tokens to the vocab rather than treating them as OOV
    :param file: path to a specific file to load if required
    :param source: source of the data to load. Should be either kbp or cloze
    :param data_type: type of data to load; train, dev or test
    :param data_dir: parent directory of the data (assumed it contains sub folders for source and type e.g. raw\\kbp\\train)
    :return: dict of preprocessed data and vocab
    """
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
    """
    Takes a dict of data still in string form and performs preprocessing (tokenisation, conversion of tokens to token
    IDs), and builds the vocab
    :param data: dict containing data in string form
    :param vocab: a prexisting vocab to use if required
    :param extend_vocab: if passing an existing vocab, add new tokens to the vocab rather than treating them as OOV
    :return:
    """
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
    """
    Converts a numpy array or nested list of numpy arrays into pure lists
    :param data: possibly nested list of numpy arrays
    :return: input in pure list form
    """
    if type(data) is np.ndarray:
        data = data.tolist()
    elif type(data) is list:
        for i, element in enumerate(data):
            data[i] = listify(element)
    return data


def listify_dict(list_dict):
    """
    Runs the listify function for data under each key in a dict
    :param list_dict:
    :return: dict containing the same data but as pure lists
    """
    for key in list_dict:
        listify(list_dict[key])
    return list_dict


def extend_dict(data_dict, new_data_dict):
    """
    Combine dicts by combining the data under each key
    :param data_dict: an existing dict to extend
    :param new_data_dict: a dict (with the same format as data_dict) to extend data_dict with
    :return: single extended dict containing data from both data_dict and new_data_dict
    """
    for key in data_dict:
        data_dict[key].extend(new_data_dict[key])
    return data_dict


def bicond_reader(placeholders, vocab_size, emb_dim, drop_keep_prob=1.0):
    """
    Builds the tensorflow graph for a bidirectional LSTM conditional reader for question answering
    :param placeholders: dict containing tensorflow placeholders
    :param vocab_size: size of the vocab of the data to be used with this model
    :param emb_dim: word embedding dimension
    :param drop_keep_prob: keep probability for dropout
    :return: score (unscaled logits), loss and prediction (normalised logits (softmax)) tensorflow ops
    """
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

        output = tf.concat(axis=1, values=[states[0][1], states[1][1]])  # [batch_size, 2*emb_dim]

        # squish back into emb_dim num dimensions
        output = tf.contrib.layers.linear(output, emb_dim)  # [batch_size, emb_dim]

        # batch matrix multiplication to get per-candidate scores
        scores = tf.einsum('bid,bcd->bc', tf.expand_dims(output, 1), candidates_embedded)  # [batch_size, 1, emb_dim], [batch_size, max_num_cands, emb_dim] -> [batch_size, max_num_cands]

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
    """
    Performs the same function as the bicond_reader function but works on inputs that have already been embedded by the
    embedding lookup op
    :param placeholders: dict containing tensorflow placeholders
    :param inputs_embedded: Inputs to the reader after already having passed through the embedding lookup ops
    :param emb_dim: word embedding dimension
    :param drop_keep_prob: keep probability for dropout
    :return: score (unscaled logits), loss and prediction (normalised logits (softmax)) tensorflow ops
    """
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
    """
    Bi-directional LSTM reader that works on pre embedded inputs. Returns a 2d output for either 'accept' or 'reject'
    :param placeholders: dict containing tensorflow placeholders
    :param inputs_embedded: Inputs to the reader after already having passed through the embedding lookup ops
    :param emb_dim: word embedding dimension
    :param drop_keep_prob: keep probability for dropout
    :return: unscaled score, selection (1 or 0 for accept or reject) and normalised prediction tensorflow ops
    """
    question_embedded = inputs_embedded['question_embedded']

    with tf.variable_scope('question_reader') as scope:
        outputs, states = reader(question_embedded, placeholders['question_lengths'], emb_dim,
                                 scope=scope, drop_keep_prob=drop_keep_prob)

    # Take final states and concatenate
    output_fw = states[0][1]  # [batch_size, emb_dim]
    output_bw = states[1][1]  # [batch_size, emb_dim]

    output = tf.concat([output_fw, output_bw], 1)  # [batch_size, 2*emb_dim]

    # Linear layer to transform from 2*emb_dim to 2 (select or not select)
    scores = tf.layers.dense(output, 2)  # [batch_size, 2]

    # Get selection
    selection = tf.argmax(scores, axis=1)  # [batch_size, 1]

    predict = tf.nn.softmax(scores)

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
    """
    Returns some data, value, as a tensorflow bytes feature for writing to TFRecords files
    :param value: some data element
    :return: tensorflow bytes feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecords(feed_dict_list, filename):
    """
    Writes data in a list of feed dicts to a TFRecords file, filename
    :param feed_dict_list: list of data feed dicts (batch size 1)
    :param filename: filename (with path if not saving to current directory) (without .tfrecords extension) to save as
    """
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
    """
    Takes a list of feed dicts and splits it up into chunks and then saves each chunk as a separate TFRecords file
    :param feed_dict_list: list of data feed dicts (batch size 1)
    :param base_filename: filename (with path if not saving to current directory) (without .tfrecords extension) to
                          save as (numbers will be appended to the filename to make each filename unique)
    :param examples_per_file: number of examples to save to each TFRecords file
    """
    filename_counter = 0
    for examples in chunker(feed_dict_list, examples_per_file):
        filename = base_filename + '_' + str(filename_counter)
        write_to_tfrecords(examples, filename)
        filename_counter += 1


def chunker(seq, size):
    """
    Splits a list into chunks for iterating over
    :param seq: list of data
    :param size: size of chunks to divide seq into
    :return: a generator that returns each chunk when iterating over
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_sizes_dict(dict_list):
    """
    Get the size of the arrays under each key in a list of dicts of the same format, padded to the same size, and return
    as dict
    :param dict_list: list of dicts containing padded data
    :return: dict of sizes
    """
    data_dict = dict_list[0]

    sizes_dict = {}
    for key in data_dict.keys():
        sizes_dict[key.op.name] = data_dict[key][0].shape

    return sizes_dict


def pad_list_of_dicts(list_of_dicts, dict_of_shapes):
    """
    Pad a list of dicts with zeros
    :param list_of_dicts: list of batch dicts
    :param dict_of_shapes: dict containing shapes to pad to
    :return: padded list of dicts
    """
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
    """
    For a list of sizes dict return a dict of max sizes in each dimension for each dict key
    :param sizes_list: list of sizes dicts
    :return: dict of max sizes
    """
    max_sizes_dict = {}
    for key in sizes_list[0].keys():
        # Create list of sizes for key
        key_sizes = [sizes[key] for sizes in sizes_list]

        # Get max size
        max_sizes_dict[key] = tuple(np.max(np.array(key_sizes), axis=0))
    return max_sizes_dict


def save_obj(obj, name):
    """
    Save an object as a .pkl file
    :param obj: Some object to save
    :param name: Filename (including path if required) to save as
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Load a previously saved object
    :param name: Filename (including path if required) to load from
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def build_tfrecord_dataset(in_path='data\\raw\\', out_path='data\\tfrecords\\', examples_per_file=1024):
    """
    Converts a raw data set in .json format to a preprocessed data set in .tfrecords format. Assumes a directory
    structure in the in_path of data source (kbp or cloze) then data type (train, dev or test)
    e.g. in_path/kbp/train should contain all the kbp train examples in .json files.
    Saves to the same directory structure in out_path and also saves vocab and shapes as a .pkl file.
    Splits the tfrecords files to only contain a maximum of examples_per_file examples per file.
    """
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

    # Delete KBP training data from memory
    kbp_train_data = None

    cloze_train_data, vocab = load_data(placeholders, 1, vocab=vocab, extend_vocab=True, source='cloze',
                                        data_type='train', data_dir=in_path)  # TODO - appears to not load the whole of the data (loads the same amount as for kbp so maybe reaching a limit for that as well?), maybe try splitting jsons before hand

    # Pad data
    cloze_train_data = pad_list_of_dicts(cloze_train_data, shapes)

    # Save cloze training data as tfrecords
    write_to_tfrecords_in_chunks(cloze_train_data, out_path + 'cloze\\train\\train_cloze', examples_per_file)

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


def build_categorised_tfrecords_dataset(in_path='data\\raw\\', out_path='data\\tfrecords\\', examples_per_file=1024):
    """
    Works in the same way as the build_tfrecords_dataset function but for the cloze train data it saves separate
    clusters in their own folder for use with the cluster based sampler. Clusters are computed at this stage rather than
    at runtime of the model
    """
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

    # First load training data and build vocab  TODO - if wanting to use levenshtein distance then will have to make deeper modifications to the loading function here
    kbp_train_data, kbp_vocab = load_data(placeholders, 1, source='kbp', data_type='train', data_dir=in_path)

    cloze_train_data, vocab = load_data(placeholders, 1, vocab=kbp_vocab, extend_vocab=True, source='cloze',
                                        data_type='train',
                                        data_dir=in_path)  # TODO - appears to not load the whole of the data (loads the same amount as for kbp so maybe reaching a limit for that as well?), maybe try splitting jsons before hand

    # COMPUTE DIFFERENT CLOZE GROUPS (GET INDICES AND THEN USE INDICES TO ADD TO)
    group_indices = [[], [], []]

    # These are redundant assignments but allow the comments on what each group is
    group_indices[0] = []  # Cloze question shorter than 30 tokens
    group_indices[1] = []  # Cloze question shorter than 20 tokens
    group_indices[2] = []  # Not in any other group

    n_groups = 3

    # Build list of KBP answers
    kbp_answers = []
    for example in kbp_train_data:
        kbp_answers.append(example[placeholders['answers']][0])

    for i, cloze_example in enumerate(cloze_train_data):
        # Decide whether to add to group 0
        question = cloze_example[placeholders['question']][0]
        if check_contains_less_than_n_tokens(question, 30):
            group_indices[0].append(i)

        # Decide whether to add to group 1
        if check_contains_less_than_n_tokens(question, 20):
            group_indices[1].append(i)


    group_indices[-1] = list(set(range(len(cloze_train_data))) - set(group_indices[0]) - set(group_indices[1]))

    for i in range(n_groups):
        cloze_train_data_group = [cloze_train_data[x] for x in group_indices[i]]

        # Pad data
        cloze_train_data_group = pad_list_of_dicts(cloze_train_data_group, shapes)

        # Save cloze training data as tfrecords
        # Create folder if required
        directory = out_path + 'cloze\\train\\group_' + str(i)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write
        write_to_tfrecords_in_chunks(cloze_train_data_group, directory + '\\train_cloze_group_' + str(i), examples_per_file)

    # Delete KBP training data from memory
    kbp_train_data = None

    # Delete cloze training data from memory
    cloze_train_data = None

    # Delete
    data = None


def create_dict_of_filename_lists(data_dir):
    """
    For a given directory form a dict of dicts/lists containing all subdirectories as keys and filenames in sub
    directories as lists
    :param data_dir: directory to form dict of filename lists
    :return: dict of filename lists
    """
    # Get folder names in data_dir to use as dict keys
    sub_dirs = [name for name in os.listdir(data_dir) if os.path.isdir(data_dir + name)]

    # Get tfrecords filenames
    tfrecords_filenames = [data_dir + file for file in os.listdir(data_dir) if file.endswith('.tfrecords')]

    # Fill dict with lists of filenames (or dicts if more folders)
    if sub_dirs:
        dict_of_filename_lists = {}
        if tfrecords_filenames:
            dict_of_filename_lists['files'] = tfrecords_filenames
        for sub_dir in sub_dirs:
            dict_of_filename_lists[sub_dir] = create_dict_of_filename_lists(data_dir + sub_dir + '/')
    else:
        dict_of_filename_lists = tfrecords_filenames

    return dict_of_filename_lists


def stack_array_lists_in_dict(dict_of_lists):
    """
    For a dict of lists of arrays (of compatible size), stack all the arrays to form a dict of arrays
    :param dict_of_lists: dict of lists of arrays
    :return: dict of arrays
    """
    for key in dict_of_lists.keys():
        if dict_of_lists[key][0].shape is not ():
            dict_of_lists[key] = np.stack(dict_of_lists[key])
        else:
            dict_of_lists[key] = np.array(dict_of_lists[key])
    return dict_of_lists


def vector_to_distribution(vector):
    """
    Take a vector, clamp negative values to zero and then normalise such that it represents a multinomial distribution
    :param vector: some vector (list or array)
    :return: distrubution with same shape as vector
    """
    distribution = vector
    distribution[distribution < 0] = 0
    distribution = distribution / np.sum(distribution)

    return distribution


def check_contains_less_than_n_not_in_vocab(q_or_s, n, vocab):
    """
    For a given Cloze question or support in preprocessed form, check if it contains less than n tokens not contained in
    the vocab
    :param q_or_s: question or support in preprocessed form
    :param n: number
    :param vocab: vocab object
    :return: check - logical
    """
    x = q_or_s

    tokens = set(x.flatten())
    vocab_tokens = set(range(vocab.next_pos))

    check = len(tokens - vocab_tokens) < n

    return check


def check_contains_n_tokens_from_kbp_question(question, n, kbp_train_data, placeholders):
    """
    For a given Cloze question in preprocessed form, check whether it contains at least n tokens that appear in any
    single kbp question
    :param question: Cloze question in preprocessed form
    :param n: number
    :param kbp_train_data: full kbp train data set in preprocessed form
    :param placeholders: dict of tensorflow placeholders
    :return: check - logical
    """
    check = False
    # Loop through kbp train data
    for kbp_example in kbp_train_data:
        kbp_question = kbp_example[placeholders['question']][0]
        if len(set(question).intersection(kbp_question)) >= n:
            check = True
            break

    return check


def check_contains_less_than_n_tokens(token_indices, n):
    """
    For some sequence of tokens (represented as indices) check that there are less than n unique tokens present
    :param token_indices: sequence of token indices (to reference vocab)
    :param n: number
    :return: check - logical
    """
    # Get number of tokens
    n_tokens = len(token_indices[token_indices != 0])

    # Check
    check = n_tokens < n

    return check


def compute_mean_levenshtein_distance(support, kbp_data, vocab, placeholders):
    """
    Compute mean levenshtein distance between a Cloze support example and all KBP supports
    :param support: Cloze support in preprocessed form
    :param kbp_data: full KBP data set in preprocessed form
    :param vocab: vocab object
    :param placeholders: dict of tensorflow placeholders
    :return: mean distance
    """
    distance_list = []
    for kbp_example in kbp_data:
        kbp_support = kbp_example[placeholders['support']]
        for kbp_support_example in [x for x in list(kbp_support) if x[0] is not 0]:
            # Convert to strings
            cloze_string = reconstruct_string_from_indices(support, vocab)
            kbp_string = reconstruct_string_from_indices(kbp_support_example, vocab)

            # Compute distance
            dist = lsn.distance(cloze_string, kbp_string)

            # Append to list
            distance_list.append(dist)

    # Compute mean
    mean_dist = np.mean(distance_list)
    return mean_dist


def reconstruct_string_from_indices(indices, vocab):
    """
    For a given sequence of word indices, reconstruct the original string using vocab
    :param indices: sequence of word indices
    :param vocab: vocab object
    :return: reconstructed string
    """
    reconstructed_string = ''
    for word_id in indices[:-1]:
        reconstructed_string += vocab.id2sym[word_id] + ' '

    reconstructed_string += vocab.id2sym[indices[-1]]

    return reconstructed_string


if __name__ is '__main__':
    print('test')
