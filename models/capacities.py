from preproc.vocab import Vocab
from preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
import tensorflow as tf
import numpy as np
import json


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
    # TODO - This is probably using a lot more memory than it needs to by making multiple copies
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


def load_data(placeholders, batch_size=1, vocab=None, extend_vocab=False, file=None, data='kbp', type='train', data_dir='data\\'):
    #train_data = dummy_data()
    if file is None:
        if data == 'kbp':
            if type == 'train':
                train_data_pos = full_data(data_dir + 'kbpLocal_train_pos.json')
                train_data_neg = full_data(data_dir + 'kbpLocal_train_neg.json')

                list_of_dicts = [train_data_pos, train_data_neg]
            elif type == 'dev':
                dev_data_pos = full_data(data_dir + 'kbpLocal_dev_pos.json')
                dev_data_neg = full_data(data_dir + 'kbpLocal_dev_neg.json')

                list_of_dicts = [dev_data_pos, dev_data_neg]
            elif type == 'test':
                test_data_pos = full_data(data_dir + 'kbpLocal_test_pos.json')
                test_data_neg = full_data(data_dir + 'kbpLocal_test_neg.json')

                list_of_dicts = [test_data_pos, test_data_neg]
            else:
                raise Exception('type must be train, dev or test')
        elif data == 'cloze':
            if type == 'train':
                train_data_pos = full_data(data_dir + 'clozeLocal_train_pos.json')
                train_data_neg = full_data(data_dir + 'clozeLocal_train_neg.json')
                # train_data_002 = full_data('data\\clozeLocal_train-002.json') TODO - sort out loading of this, currently get memory error. Maybe use queues and tfrecords files?

                list_of_dicts = [train_data_pos, train_data_neg]
            elif type == 'dev':
                dev_data_pos = full_data(data_dir + 'clozeLocal_dev_pos.json')
                dev_data_neg = full_data(data_dir + 'clozeLocal_dev_neg.json')
                dev_data = full_data(data_dir + 'clozeLocal_dev.json')

                list_of_dicts = [dev_data_pos, dev_data_neg, dev_data]
            elif type == 'test':
                test_data_pos = full_data(data_dir + 'clozeLocal_test_pos.json')
                test_data_neg = full_data(data_dir + 'clozeLocal_test_neg.json')
                test_data = full_data(data_dir + 'clozeLocal_test.json')

                list_of_dicts = [test_data_pos, test_data_neg, test_data]
            else:
                raise Exception('type must be train, dev or test')
        else:
            raise Exception('data must be kbp or cloze')

        data = combine_data(list_of_dicts)
    else:
        data = full_data(file)

    data, vocab = prepare_data(data, vocab=vocab, extend_vocab=extend_vocab)
    # TODO get newer version of get_feed_dicts working
    #train_feed_dicts = get_feed_dicts(data, placeholders, batch_size=batch_size, bucket_order=None,
    #                                     bucket_structure=None)
    feed_dicts = get_feed_dicts_old(data, placeholders, batch_size=batch_size, bucket_order=None, bucket_structure=None)
    return feed_dicts, vocab


def load_data_dicts(vocab=None, extend_vocab=False, file=None, data='kbp', type='train', data_dir='data\\'):
    if file is None:
        if data == 'kbp':
            if type == 'train':
                train_data_pos = full_data(data_dir + 'kbpLocal_train_pos.json')
                train_data_neg = full_data(data_dir + 'kbpLocal_train_neg.json')

                list_of_dicts = [train_data_pos, train_data_neg]
            elif type == 'dev':
                dev_data_pos = full_data(data_dir + 'kbpLocal_dev_pos.json')
                dev_data_neg = full_data(data_dir + 'kbpLocal_dev_neg.json')

                list_of_dicts = [dev_data_pos, dev_data_neg]
            elif type == 'test':
                test_data_pos = full_data(data_dir + 'kbpLocal_test_pos.json')
                test_data_neg = full_data(data_dir + 'kbpLocal_test_neg.json')

                list_of_dicts = [test_data_pos, test_data_neg]
            else:
                raise Exception('type must be train, dev or test')
        elif data == 'cloze':
            if type == 'train':
                train_data_pos = full_data(data_dir + 'clozeLocal_train_pos.json')
                train_data_neg = full_data(data_dir + 'clozeLocal_train_neg.json')
                # train_data_002 = full_data('data\\clozeLocal_train-002.json') TODO - sort out loading of this, currently get memory error. Maybe use queues and tfrecords files?

                list_of_dicts = [train_data_pos, train_data_neg]
            elif type == 'dev':
                dev_data_pos = full_data(data_dir + 'clozeLocal_dev_pos.json')
                dev_data_neg = full_data(data_dir + 'clozeLocal_dev_neg.json')
                dev_data = full_data(data_dir + 'clozeLocal_dev.json')

                list_of_dicts = [dev_data_pos, dev_data_neg, dev_data]
            elif type == 'test':
                test_data_pos = full_data(data_dir + 'clozeLocal_test_pos.json')
                test_data_neg = full_data(data_dir + 'clozeLocal_test_neg.json')
                test_data = full_data(data_dir + 'clozeLocal_test.json')

                list_of_dicts = [test_data_pos, test_data_neg, test_data]
            else:
                raise Exception('type must be train, dev or test')
        else:
            raise Exception('data must be kbp or cloze')

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


def extend_dict(data_dict, new_data_dict):
    for key in data_dict:
        data_dict[key].extend(new_data_dict[key])
    return data_dict
