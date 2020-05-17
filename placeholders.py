import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

a = tf.placeholder(tf.float32)

b = a * 2

with tf.Session() as sess:
    print(sess.run(b, feed_dict={a: 3.5}))

dictionary = {a: [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}
with tf.Session() as sess:
    print(sess.run(b, feed_dict=dictionary))
