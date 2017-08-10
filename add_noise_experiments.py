import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import numpy as np

batch_size = 1
seq_len = 10
emb_dim = 20
rnn_size = 20

# Create base graph
x = tf.placeholder('float', shape=[None, seq_len, emb_dim])

lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
rnn_output = rnn_outputs[:, -1, :]

h1 = tf.layers.dense(rnn_output, 20, tf.nn.relu)  # TODO - test with different graph i.e add rnn or something
h2 = tf.layers.dense(h1, 2, tf.nn.relu)

logits = tf.nn.softmax(h2)

selection = tf.argmax(logits, axis=1)

# Create Delta noise placeholders and get shape of required noise tensors to be fed
placeholder_shapes = []
delta_placeholders = []
trainable_variables = tf.trainable_variables()  # TODO - test only adding placeholders to variables in specific scope(s)
gamma_update_op = None
for var in trainable_variables:
    # Add summary
    tf.summary.histogram(var.name, var)

    # Get shape
    var_shape = var.shape
    placeholder_shapes.append(var_shape)

    # Create placeholder
    delta_placeholder = tf.placeholder('float', shape=var_shape)
    delta_placeholders.append(delta_placeholder)

    # Get ops from forward walk from var
    fw_ops = ge.get_forward_walk_ops(var.op.outputs)

    not_types = ['Assign', 'Identity', 'HistogramSummary', 'Enter']

    next_op = None

    # Select the correct op from the forward walk to connect to
    for fw_op in fw_ops:
        if fw_op.type not in not_types:
            next_op = fw_op
            break

    if next_op is None:
        raise ValueError('No suitable next op found to connect to. Try looking at the graph or full list of forward ops')

    # Add placeholder and variable
    add_op = tf.add(var, delta_placeholder)  # TODO - might be neater if these were created in the same scope as the variable; also might solve issue with connecting add ops within while loop

    # Connect add_op output to next op input
    # Create subgraph 1 (outputs)
    sgv0 = ge.sgv(add_op.op)

    # Create subgraph 2 (inputs)
    sgv1 = ge.sgv(next_op).remap_inputs([1])

    # Connect
    ge.connect(sgv0, sgv1)  # TODO - sort out error with tf.while loops; may not be possible: try the assign_add method first

    # Define parameter update ops
    update = 0.01*delta_placeholder
    new_update_op = tf.assign_sub(var, update)

    if gamma_update_op is not None:
        gamma_update_op = tf.group(new_update_op, gamma_update_op)
    else:
        gamma_update_op = new_update_op

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('results\\add_noise_experiments\\', sess.graph)

summary_writer.add_graph(sess.graph)

for i in range(10):
    feed_dict = {x: np.random.normal(size=(batch_size, seq_len, emb_dim))}
    for delta_placeholder, var_shape in zip(delta_placeholders, placeholder_shapes):
        feed_dict.update({delta_placeholder: np.random.normal(size=var_shape)})

    print('FIRST TRAINABLE VARIABLE')
    print(sess.run(trainable_variables[0]))

    select, summary = sess.run([selection, merged], feed_dict=feed_dict)
    summary_writer.add_summary(summary, i)

    sess.run(gamma_update_op, feed_dict=feed_dict)

    print('FEED DICT')
    print(feed_dict)
    print('SELECT')
    print(select)

