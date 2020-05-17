import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant([2.0])
    b = tf.nn.sigmoid(a)

with tf.Session(graph=graph1) as sess:
    print(sess.run(b))
