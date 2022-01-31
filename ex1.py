import tensorflow as tf
import numpy as np

const = tf.Variable(2.0, name="const")
b = tf.Variable(2.0, name="b")
c = tf.Variable(1.0, name="c")

d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

print(f"Variable a is {a.numpy()}")

b = tf.Variable(np.arange(0, 10), name='b')
d = tf.cast(b, tf.float32) + c
print(d)

a = tf.multiply(d, e, name='a')

print(f"Variable a is {a.numpy()}")

b[1].assign(10)
print(b)

d = tf.cast(b, tf.float32) + c
a = tf.multiply(d, e, name='a')

print(f"Variable a is {a.numpy()}")

print(b)

f = b[2:2]
print(f)

f = b[2:3]
print(f)

f = b[2:4]
print(f)

f = b[2:5]
print(f)

