# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class ExtractModel(object):
    def __init__(self, name="ExtractionModel", extract_size=64, label_classes=2):
        self.name = name
        self.extract_size = extract_size
        self.label_classes = label_classes

    def build(self, data, reuse=None, training=False):
        with tf.variable_scope(self.name, reuse=reuse):
            wave = tf.cast(data["wave"], tf.float32)
            labels = tf.cast(data["labels"], tf.int32)
            with tf.variable_scope("cnn", reuse=reuse):
                """extract feature from raw wave."""
                output = tf.layers.conv1d(wave, filters=16, kernel_size=7, strides=1, padding='valid',
                                          activation=tf.nn.relu,  name="conv_layer_1")
                output = tf.layers.max_pooling1d(output, pool_size=3, strides=2, padding='same',
                                                 name="pool_layer_1")
                output = tf.layers.batch_normalization(output, training=training, name="bn_layer_1")
                output = tf.layers.conv1d(output, filters=32, kernel_size=5, strides=1, padding='valid',
                                          activation=tf.nn.relu, name="conv_layer_2")
                output = tf.layers.max_pooling1d(output, pool_size=3, strides=2, padding='same',
                                                 name="pool_layer_2")
                output = tf.layers.batch_normalization(output, training=training, name="bn_layer_2")
                output = tf.layers.conv1d(output, filters=self.extract_size, kernel_size=3, strides=1, padding='valid',
                                          activation=tf.nn.relu, name="conv_layer_3")
                output = tf.layers.max_pooling1d(output, pool_size=3, strides=2, padding='same',
                                                 name="pool_layer_3")
                output = tf.layers.batch_normalization(output, training=training, name="bn_layer_3")
                # TODO extract feature from raw wave.
            with tf.variable_scope("rnn", reuse=reuse):
                """express output of CNN on time step dimension."""
                fw_cells = list()
                bw_cells = list()
                rnn_layer_num = 2
                keep_prob = 0.5
                for _ in range(rnn_layer_num):
                    # Define cells
                    fw_cell = rnn.BasicLSTMCell(self.extract_size, forget_bias=1.0)
                    bw_cell = rnn.BasicLSTMCell(self.extract_size, forget_bias=1.0)
                    # Drop out in case of over-fitting.
                    fw_cell = rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob)
                    bw_cell = rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob)
                    # Stack same cells.
                    fw_cells.append(fw_cell)
                    bw_cells.append(bw_cell)
                output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                    fw_cells,
                    bw_cells,
                    output,
                    dtype=tf.float32
                )
                # TODO express output of CNN on time step dimension.
            with tf.variable_scope("fully_connected", reuse=reuse):
                """classify output of RNN by a fully-connected network."""
                output = tf.layers.dense(inputs=output, units=32, activation=tf.nn.relu, name="dense_layer_1")
                output = tf.layers.batch_normalization(output, training=training, name="bn_layer_1")
                output = tf.layers.dense(inputs=output, units=16, activation=tf.nn.relu, name="dense_layer_2")
                output = tf.layers.batch_normalization(output, training=training, name="bn_layer_2")
                output = tf.layers.dense(inputs=output, units=8, activation=tf.nn.relu, name="dense_layer_3")
                output = tf.layers.batch_normalization(output, training=training, name="bn_layer_3")
                # TODO classify output of RNN by a fully-connected network.
            logits = tf.layers.dense(inputs=output, units=self.label_classes, activation=tf.nn.softmax,
                                     name="output_layer")
            predictions = tf.argmax(logits, axis=-1)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            acc, acc_op = tf.metrics.accuracy(labels, predictions)
            rec, rec_op = tf.metrics.recall(labels, predictions)
            pre, pre_op = tf.metrics.precision(labels, predictions)
            f1_score = tf.divide(2 * tf.multiply(rec_op, pre_op), tf.add(rec_op, pre_op))
            return {"predictions": predictions, "labels": labels, "loss": loss,
                    "accuracy": acc_op, "recall": rec_op, "precision": pre_op, "f1_score": f1_score}
