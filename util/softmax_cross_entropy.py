import tensorflow as tf

logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

# step1:do softmax

print(logits)

y = tf.nn.softmax(logits)

# true label
y_ = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

# step2:do cross_entropy

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

with tf.Session() as sess:
    softmax = sess.run(y)
    c_e = sess.run(cross_entropy)

    print(softmax)
    print(c_e)


