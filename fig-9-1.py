import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
print(f)

with tf.Session () as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

