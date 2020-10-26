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
        self.text_b = tf.placeholder(tf.int32, [None, seq_length], name="text_b")
        self.y = tf.placeholder(tf.int32, [None, class_num], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding 层，使用随机初始化，论文是预训练好的
        with tf.name_scope("embedding"):
            self.vocab_matrix = tf.get_variable("vocab_matrix", shape=[vocabulary_size, embedding_size],
                                                initializer=tf.constant_initializer(np.array(init_embedding_up)))
            self.text_a_embedding = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
            self.text_b_embedding = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)

        #  Input Encoding
        with tf.name_scope("Input_encoding"):
            a_bar = self.cnn_inference(self.text_a_embedding, hidden_num, "Input_Encoding/cnn")
            b_bar = self.cnn_inference(self.text_b_embedding, hidden_num, "Input_Encoding/cnn")

        #     local inference modeling
        with tf.name_scope("local_inference_modeling"):
            #  计算a_bar 与 b_bar 相似度
            with tf.name_scope("word_similarity"):
                attention_weights = tf.matmul(a_bar, tf.transpose(b_bar, [0, 2, 1]))
                attention_a = tf.nn.softmax(attention_weights)
                attention_b = tf.nn.softmax(tf.transpose(attention_weights, [0, 2, 1]))
                a_hat = tf.matmul(attention_a, b_bar)
                b_hat = tf.matmul(attention_b, a_bar)

            # 计算m_a,m_b
            with tf.name_scope("compute_m_a/m_b"):
                a_diff = tf.subtract(a_bar, a_hat)
                a_mul = tf.multiply(a_bar, a_hat)

                b_diff = tf.subtract(b_bar, b_hat)
                b_mul = tf.multiply(b_bar, b_hat)

                self.m_a = tf.concat([a_bar, a_hat, a_diff, a_mul], axis=2)
                self.m_b = tf.concat([b_bar, b_hat, b_diff, b_mul], axis=2)

        with tf.name_scope("inference_composition"):
            v_a = self.cnn_inference(self.m_a, hidden_num, "inference_composition/cnn")
            v_b = self.cnn_inference(self.m_b, hidden_num, "inference_composition/cnn")
            # average pool and max_pool
            v_a_averge = tf.reduce_mean(v_a, axis=1)
            v_b_averge = tf.reduce_mean(v_b, axis=1)
            v_a_max = tf.reduce_max(v_a, axis=1)
            v_b_max = tf.reduce_max(v_b, axis=1)

            v = tf.concat([v_a_averge, v_a_max, v_b_averge, v_b_max], axis=1)

        with tf.name_scope("output"):
            initializer = tf.truncated_normal_initializer(0.0, 0.1)
            with tf.variable_scope("feed_forward_layer1"):
                outputs = tf.layers.dense(v, 100, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope("feed_forward_layer2"):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                self.logits = tf.layers.dense(outputs, 2, kernel_initializer=initializer)
            self.score = tf.nn.softmax(self.logits, name="score")
            self.prediction = tf.argmax(self.score, 1, name="prediction")

        with tf.name_scope("cost"):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            self.cost = tf.reduce_mean(self.cost)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = l2_loss + self.cost
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32)
        )

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
            conv1_weights = tf.get_variable(
                "weight", [3, inputs_shape[2], 1, num_units],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            conv1_biaes = tf.get_variable("bias", [num_units], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(self.pad_for_wide_conv(inputs), conv1_weights, strides=[1, 1, 1, 1],
                                 padding='VALID')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biaes))
            relu1 = tf.squeeze(relu1, 2)
            return relu1

    def pad_for_wide_conv(self, x):
        return tf.pad(x, np.array([[0, 0], [1, 1], [0, 0], [0, 0]]), "CONSTANT", name="pad_wide_conv")


if __name__ == '__main__':
    esim = ESIM(True, 20, 2, 10000, 300, 300, 2, 0.01, 0.0001)
