import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

""" create a graph """
graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant([2], name='constant_a')
    b = tf.constant([3], name='constant_b')
    c = tf.add(a, b)

sess = tf.Session(graph=graph1)
result = sess.run(c)
print(result)
sess.close()
