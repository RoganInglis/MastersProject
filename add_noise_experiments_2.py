import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import numpy as np

batch_size = 1
seq_len = 10
emb_dim = 20
rnn_size = 20

# Create base graph
x = tf.placeholder('float', shape=[None, seq_len, emb_dim])

with tf.variable_scope('lstm'):
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
add_ops = None
sub_ops = None
gen = (var for var in trainable_variables if 'lstm' not in var.name)

for var in gen:
    # Add summary
    tf.summary.histogram(var.name, var)

    # Get shape
    var_shape = var.shape
    placeholder_shapes.append(var_shape)

    # Create placeholder
    delta_placeholder = tf.placeholder('float', shape=var_shape)
    delta_placeholders.append(delta_placeholder)

    # Add placeholder and variable
    add_op = tf.assign_add(var, delta_placeholder)
    sub_op = tf.assign_sub(var, delta_placeholder)

    # Define parameter update ops
    update = 0.01*delta_placeholder
    new_update_op = tf.assign_sub(var, update)

    if gamma_update_op is not None:
        gamma_update_op = tf.group(new_update_op, gamma_update_op)
    else:
        gamma_update_op = new_update_op

    if add_ops is not None:
        add_ops = tf.group(add_op, add_ops)
    else:
        add_ops = add_op

    if sub_ops is not None:
        sub_ops = tf.group(sub_op, sub_ops)
    else:
        sub_ops = sub_op

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('results\\add_noise_experiments\\', sess.graph)

summary_writer.add_graph(sess.graph)

for i in range(10):
    # Create feed dict of input and variable noise
    feed_dict = {x: np.random.normal(size=(batch_size, seq_len, emb_dim))}
    for delta_placeholder, var_shape in zip(delta_placeholders, placeholder_shapes):
        feed_dict.update({delta_placeholder: np.random.normal(size=var_shape)})

    # Display variables before noise is added
    print('FIRST TRAINABLE VARIABLE BEFORE NOISE')
    print(sess.run(trainable_variables[0]))

    # Add noise
    sess.run(add_ops, feed_dict=feed_dict)

    # Display variables after noise is added
    print('FIRST TRAINABLE VARIABLE AFTER NOISE')
    print(sess.run(trainable_variables[0]))

    # Select (would select training examples for batch here)
    select, summary = sess.run([selection, merged], feed_dict=feed_dict)
    summary_writer.add_summary(summary, i)

    # Subtract noise
    sess.run(sub_ops, feed_dict=feed_dict)

    # Display variables after noise is added
    print('FIRST TRAINABLE VARIABLE AFTER SUBTRACTING NOISE')
    print(sess.run(trainable_variables[0]))

    # Perform update
    sess.run(gamma_update_op, feed_dict=feed_dict)

    print('FEED DICT')
    print(feed_dict)
    print('SELECT')
    print(select)

