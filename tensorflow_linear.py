import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.randn(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

