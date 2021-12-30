import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[1., 2.]])

# product = tf.matmul(matrix1, matrix2.)]

# 计算张量的各个维度上的元素的平均值
x = tf.constant([[1., 1.], [2., 2.]])
tf.reduce_mean(x)  # 1.5
tf.reduce_mean(x, 0)  # [1.5, 1.5]
tf.reduce_mean(x, 1)  # [1.,  2.]

# tensorflow 变量，共享变量，常量
# 变量
w: tf.Variable = tf.Variable(3, name="x")
# type 类型：tensorflow.python.ops.variables.RefVariable
print(dir(w))
print("_______________")
# 共享变量
# print(tf.get_variable(name="x"))
# 占位符
p: tf.Tensor = tf.placeholder(dtype=tf.float32, name="placeholder_example")
print(dir(p))
print("_______________")
# 常量
c: tf.Tensor = tf.constant([[3., 3.]])
# 为啥这里的typing是tf.Tensor 和 tf.constant 都是Ok的？
print(dir(c))
print("----------------")
x = tf.random_normal(shape=[2, 3])
with tf.Session() as sess:
    print(sess.run(x))

query_layer = tf.layers.dense(
    x,
    4 * 5, )
# TODO
# [2,3] -> [2,20]
print(query_layer.shape)
