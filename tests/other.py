import tensorflow as tf
import numpy as np

batch_size = 1

# Create base graph
x = tf.placeholder('float', shape=[None, 10])

h1 = tf.layers.dense(x, 20, tf.nn.relu)
h2 = tf.layers.dense(h1, 2, tf.nn.relu)

logits = tf.nn.softmax(h2)

selection = tf.argmax(logits, axis=1)

# Create Delta noise placeholders and get shape of required noise tensors to be fed
placeholder_shapes = []
delta_placeholders = []
add_ops = []
trainable_variables = tf.trainable_variables()
all_add_ops = None
for var in trainable_variables:
    # Add summary
    tf.summary.histogram(var.name, var)

    # Get shape
    var_shape = var.shape
    placeholder_shapes.append(var_shape)

    # Create placeholder
    placeholder = tf.placeholder('float', shape=var_shape)
    delta_placeholders.append(placeholder)

    # Add placeholder and variable
    add_op = tf.assign_add(var, placeholder) # TODO - Will work with assign_add but not the most elegant solution as we will then either have to copy the variable or subtract the noise again to get the original variable for the update

    if all_add_ops is None:
        all_add_ops = add_op
    else:
        all_add_ops = tf.group(add_op, all_add_ops)

    """
    # Make SGVs
    sgv1 = tf.contrib.graph_editor.sgv(var.graph)
    sgv2 = tf.contrib.graph_editor.sgv(add_op)

    tf.contrib.graph_editor.swap_outputs(sgv1, sgv2)
    """

# Define parameter update ops

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('results\\other\\', sess.graph)

feed_dict = {x: np.random.normal(size=(batch_size, 10))}
for placeholder, var_shape in zip(delta_placeholders, placeholder_shapes):
    feed_dict.update({placeholder: np.random.normal(size=var_shape)})

# Print first trainable variable
print('FIRST TRAINABLE VARIABLE PRE')
print(sess.run(trainable_variables[0]))

# Print logits
print('LOGITS PRE')
print(sess.run(logits, feed_dict=feed_dict))

# Run add ops
sess.run(all_add_ops, feed_dict=feed_dict)

# Print first trainable variable
print('FIRST TRAINABLE VARIABLE POST')
print(sess.run(trainable_variables[0]))

# Print logits
print('LOGITS POST')
print(sess.run(logits, feed_dict=feed_dict))

select, summary = sess.run([selection, merged], feed_dict=feed_dict)
summary_writer.add_summary(summary)

print(feed_dict)
print(select)

"""
delta = tf.random_normal([2, 3], seed=1234)
gamma = tf.Variable(initial_value=tf.constant(2., shape=[2, 3]), name='variable_1')
const = tf.random_normal([1], seed=1234)

#gamma_delta_sum = tf.add(gamma, delta)
trainable_variables = tf.trainable_variables()

# Create list of rand tensors to add to trainable variables and define updates
const_list = []
for var in trainable_variables:
    const = tf.random_normal([1], seed=1234)
    const_list.append(const)


print(trainable_variables[0].shape)

update = tf.multiply(0.01*const, delta)
variable_update = tf.assign_sub(gamma, update)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(tf.trainable_variables())
"""

"""
Need to 
- add random noise, Delta, to parameters during batch selection
- use same random noise for variable update

Do using placeholders and feed numpy random tensors?
"""