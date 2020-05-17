import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

graph1 = tf.Graph()

with graph1.as_default():
    scalar = tf.constant([2])
    vector = tf.constant([1, 2, 3])
    matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor = tf.constant(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

with tf.Session(graph=graph1) as sess:
    result = sess.run(scalar)
    print(result)
    result = sess.run(vector)
    print(result)
    result = sess.run(matrix)
    print(result)
    result = sess.run(tensor)
    print(result)

########################################################

print(scalar.shape)
print(vector.shape)
print(matrix.shape)
print(tensor.shape)

#########################################################
"""Matrix Addition"""

graph2 = tf.Graph()
with graph2.as_default():
    m1 = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    m2 = tf.constant([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    m_sum = tf.add(m1, m2)
    m_sum2 = m1 + m2

with tf.Session(graph=graph2) as sess:
    result = sess.run(m_sum)
    print(result)
    result = sess.run(m_sum2)
    print(result)

###################################################
"""matrix multiplication"""

graph3 = tf.Graph()
with graph3.as_default():
    m1 = tf.constant([[2, 2], [3, 3]])
    m2 = tf.constant([[1, 0], [0, 1]])
    m_mul = tf.matmul(m1, m2)

with tf.Session(graph=graph3) as sess:
    result = sess.run(m_mul)
    print(result)

#####################################################
