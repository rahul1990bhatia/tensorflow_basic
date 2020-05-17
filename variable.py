import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

v = tf.Variable(1)

update = tf.assign(v, v + 1)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(v))
    for _ in range(3):
        session.run(update)
        print(session.run(v))
