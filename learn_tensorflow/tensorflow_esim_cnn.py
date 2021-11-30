import tensorflow as tf
import numpy as np
import math


class ESIM:
    def __init__(self, is_training, init_embedding_up, seq_length, class_num, vocabulary_size, embedding_size,
                 hidden_num, l2_lambda, learning_rate):
        self.is_training = is_training
        self.class_num = class_num
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_num = hidden_num
        self.seq_length = seq_length

        # init placeholder
        self.text_a = tf.placeholder(tf.int32, [None, seq_length], name="text_a")
        # 这里的None的传入的是batch_size,这里的参数，没有batch_size
        self.text_b = tf.placeholder(tf.int32, [None, seq_length], name="text_b")
        self.y = tf.placeholder(tf.int32, [None, class_num], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding 层，使用随机初始化，论文是预训练好的
        with tf.name_scope("embedding"):
            self.vocab_matrix = tf.get_variable("vocab_matrix", shape=[vocabulary_size, embedding_size],
                                                initializer=tf.constant_initializer(np.array(init_embedding_up)))
            self.text_a_embedding = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
            #  shape = [None, seq_length,embedding_size] ?
            print("self.text_a_embedding shape is %s" % self.text_a_embedding.shape)
            self.text_b_embedding = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)
            print("self.text_b_embedding shape is %s" % self.text_b_embedding.shape)

        #  Input Encoding
        with tf.name_scope("Input_encoding"):
            a_bar = self.cnn_inference(self.text_a_embedding, hidden_num, "Input_Encoding/cnn")
            b_bar = self.cnn_inference(self.text_b_embedding, hidden_num, "Input_Encoding/cnn")
        #   a_bar/b_bar.shape = [None,seq_length,hidden_num]
        print("a_bar shape is %s" % a_bar.shape)
        print("b_bar shape is %s" % b_bar.shape)

        #     local inference modeling
        with tf.name_scope("local_inference_modeling"):
            #  计算a_bar 与 b_bar 相似度
            with tf.name_scope("word_similarity"):
                attention_weights = tf.matmul(a_bar, tf.transpose(b_bar, [0, 2, 1]))
                # attention_weights.shape = [None,seq_length,seq_length]
                print("attention_weights is %s" % attention_weights)
                attention_a = tf.nn.softmax(attention_weights)
                # attention_a.shape  = [None,seq_length,seq_length]
                attention_b = tf.nn.softmax(tf.transpose(attention_weights, [0, 2, 1]))
                print("attention_b shape is %s" % attention_b.shape)
                # attention_a.shape  = [None,seq_length,seq_length]
                a_hat = tf.matmul(attention_a, b_bar)
                # [None,seq_length,seq_length] * [None,seq_length,hidden_num] =  [None,seq_length,hidden_num]
                b_hat = tf.matmul(attention_b, a_bar)
                print("a_hat shape is %s" % a_hat.shape)
                # a_hat.shape,b_hat.shape= [None,seq_length,hidden_num]

            # 计算m_a,m_b
            with tf.name_scope("compute_m_a/m_b"):
                a_diff = tf.subtract(a_bar, a_hat)
                # 对应位相减？
                # a_diff.shape= [None,seq_length,hidden_num]
                a_mul = tf.multiply(a_bar, a_hat)
                # 按位相乘 [None,seq_length,hidden_num]
                print("a_mul.shape is %s" % a_mul.shape)

                b_diff = tf.subtract(b_bar, b_hat)
                b_mul = tf.multiply(b_bar, b_hat)

                self.m_a = tf.concat([a_bar, a_hat, a_diff, a_mul], axis=2)
                self.m_b = tf.concat([b_bar, b_hat, b_diff, b_mul], axis=2)
                # [None, seq_length,4 * hidden_num]
                print("self.m_a .shape is %s" % self.m_a.shape)

        with tf.name_scope("inference_composition"):
            v_a = self.cnn_inference(self.m_a, hidden_num, "inference_composition/cnn")
            v_b = self.cnn_inference(self.m_b, hidden_num, "inference_composition/cnn")
            print("v_a .shape is %s" % v_a.shape)
            # average pool and max_pool
            v_a_averge = tf.reduce_mean(v_a, axis=1)
            v_b_averge = tf.reduce_mean(v_b, axis=1)
            # 计算张量的各个维度上的元素的平均值.
            print("v_a_averge .shape is %s" % v_a_averge.shape)
            v_a_max = tf.reduce_max(v_a, axis=1)
            v_b_max = tf.reduce_max(v_b, axis=1)
            print("v_a_max .shape is %s" % v_a_max.shape)

            v = tf.concat([v_a_averge, v_a_max, v_b_averge, v_b_max], axis=1)
            print("v.shape is %s" % v.shape)

        with tf.name_scope("output"):
            initializer = tf.truncated_normal_initializer(0.0, 0.1)
            with tf.variable_scope("feed_forward_layer1"):
                outputs = tf.layers.dense(v, 100, tf.nn.relu, kernel_initializer=initializer)
                print("outputs.shape is %s" % outputs.shape)
            with tf.variable_scope("feed_forward_layer2"):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                print("outputs.shape is %s" % outputs.shape)
                self.logits = tf.layers.dense(outputs, 2, kernel_initializer=initializer)
                print("self.logits .shape is %s" % self.logits.shape)

            self.score = tf.nn.softmax(self.logits, name="score")
            print("self.score .shape is %s" % self.score.shape)
            self.prediction = tf.argmax(self.score, 1, name="prediction")
            print("self.prediction.shape is %s" % self.prediction.shape)

        with tf.name_scope("cost"):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            print("self.cost.shape is %s" % self.cost.shape)
            self.cost = tf.reduce_mean(self.cost)
            print("self.cost.shape is %s" % self.cost.shape)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            print("weights is %s" % weights)
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            print("l2_loss is %s" % l2_loss.shape)
            self.loss = l2_loss + self.cost
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32)
        )
        print("self.accuracy is %s" % self.accuracy.shape)

        if not is_training:
            return
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cnn_inference(self, inputs, num_units, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs = tf.expand_dims(inputs, -1)
            inputs_shape = inputs.get_shape().as_list()
            print("---------inputs_shape is %s" % inputs_shape)
            conv1_weights = tf.get_variable(
                "weight", [3, inputs_shape[2], 1, num_units],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            # conv1_weights.shape = [3, inputs_shape[2], 1, num_units]
            conv1_biaes = tf.get_variable("bias", [num_units], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(self.pad_for_wide_conv(inputs), conv1_weights, strides=[1, 1, 1, 1],
                                 padding='VALID')
            print("conv1 shape is %s" % conv1.shape)
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biaes))
            print("relu1 shape is %s" % relu1.shape)
            relu1 = tf.squeeze(relu1, 2)
            print("relu1 squeeze shape is %s" % relu1.shape)
            return relu1

    def pad_for_wide_conv(self, x):
        # input: [None, seq_length,embedding_size,1]
        # output: ?
        temp = tf.pad(x, np.array([[0, 0], [1, 1], [0, 0], [0, 0]]), "CONSTANT", name="pad_wide_conv")
        print("temp shape is %s" % temp.shape)
        return temp


if __name__ == '__main__':
    print(tf.__version__)
    assert tf.__version__ == "1.14.0"
    esim = ESIM(is_training=True, init_embedding_up=20, seq_length=2, class_num=10000, vocabulary_size=500,
                embedding_size=300, hidden_num=5,
                l2_lambda=0.01, learning_rate=0.0001)
