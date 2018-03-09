# coding=utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class ExtractModel(object):
    def __init__(self, name="ExtractionModel", extract_size=256, label_classes=2):
        self.name = name
        self.extract_size = extract_size
        self.label_classes = label_classes

    def build(self, data, reuse=None, training=False):
        with tf.variable_scope(self.name, reuse=reuse):
            wav = tf.cast(data["wav"], tf.float32)
            mu_wav = mu_law(wav)
            mu_wav = tf.expand_dims(mu_wav, axis=-1)  # expand channel dimension
            labels = tf.cast(data["label"], tf.int32)
            cnn_output = cnn(mu_wav, extract_size=self.extract_size, reuse=reuse, training=training)
            cnn_output = tf.squeeze(cnn_output, axis=1)  # squeeze time step dimension
            tf.logging.info("cnn output shape: " + str(cnn_output.get_shape()))
            # rnn_output = bi_rnn(cnn_output, extract_size=self.extract_size, reuse=reuse)
            # tf.logging.info("rnn output shape: " + str(rnn_output.get_shape()))
            dense_output = dense(cnn_output, reuse=reuse, training=training)
            tf.logging.info("dense output shape: " + str(dense_output.get_shape()))
            logits = tf.layers.dense(inputs=dense_output, units=self.label_classes, name="output_layer")
            # logits = tf.squeeze(logits, axis=1)  # squeeze time step dimension
            metric_dict = metrics(logits=logits, labels=labels)
            # metric_dict["wav"] = wav
            # metric_dict["cnn_output"] = cnn_output
            # metric_dict["rnn_output"] = rnn_output
            # metric_dict["dense_output"] = dense_output
            return metric_dict


def mu_law(x, mu=255, int8=False):
    """
    A TF implementation of Mu-Law encoding.
    Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.
    int8: Use int8 encoding.
    Returns:
    out: The Mu-Law encoded int8 data.
    """
    out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out


def cnn(inputs, extract_size, reuse=None, training=False):
    with tf.variable_scope("cnn", reuse=reuse):
        """extract feature from raw wave."""
        output = inputs
        for i in range(1, 9):
            output = tf.layers.conv1d(output, filters=2**i, kernel_size=7, strides=1, padding='same',
                                      activation=tf.nn.relu, name="conv_layer_"+str(i))
            output = tf.layers.max_pooling1d(output, pool_size=3, strides=2, padding='same',
                                             name="pool_layer_"+str(i))
            output = tf.layers.batch_normalization(output, training=training, name="bn_layer_"+str(i))
        output = tf.layers.conv1d(output, filters=extract_size, kernel_size=2, strides=1, padding='valid',
                                  activation=tf.nn.relu, name="conv_layer_6")
        output = tf.layers.batch_normalization(output, training=training, name="bn_layer_6")
        # TODO extract feature from raw wave.
    return output


def bi_rnn(inputs, extract_size, reuse=None):
    with tf.variable_scope("rnn", reuse=reuse):
        """express output of CNN on time step dimension."""
        fw_cells = list()
        bw_cells = list()
        rnn_layer_num = 2
        keep_prob = 0.5
        for _ in range(rnn_layer_num):
            # Define cells
            fw_cell = rnn.BasicLSTMCell(extract_size, forget_bias=1.0)
            bw_cell = rnn.BasicLSTMCell(extract_size, forget_bias=1.0)
            # Drop out in case of over-fitting.
            fw_cell = rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob,
                                         output_keep_prob=keep_prob)
            bw_cell = rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob,
                                         output_keep_prob=keep_prob)
            # Stack same cells.
            fw_cells.append(fw_cell)
            bw_cells.append(bw_cell)
        output, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, inputs, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        seq_len = tf.shape(output)[0]-1
        output = tf.gather_nd(output, [seq_len])
        # TODO express output of CNN on time step dimension.
    return output


def dense(inputs, reuse=None, training=False):
    with tf.variable_scope("fully_connected", reuse=reuse):
        """classify output of RNN by a fully-connected network."""
        output = tf.layers.dense(inputs=inputs, units=32, activation=tf.nn.relu, name="dense_layer_1")
        output = tf.layers.batch_normalization(output, training=training, name="bn_layer_1")
        output = tf.layers.dense(inputs=output, units=16, activation=tf.nn.relu, name="dense_layer_2")
        output = tf.layers.batch_normalization(output, training=training, name="bn_layer_2")
        output = tf.layers.dense(inputs=output, units=8, activation=tf.nn.relu, name="dense_layer_3")
        output = tf.layers.batch_normalization(output, training=training, name="bn_layer_3")
        # TODO classify output of RNN by a fully-connected network.
    return output


def metrics(logits, labels):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=-1)
    acc, acc_op = tf.metrics.accuracy(labels, predictions)
    rec, rec_op = tf.metrics.recall(labels, predictions)
    pre, pre_op = tf.metrics.precision(labels, predictions)
    f1_score = tf.divide(2 * tf.multiply(rec_op, pre_op), tf.add(rec_op, pre_op))
    return {"logits": logits, "predictions": predictions, "labels": labels, "loss": loss,
            "accuracy": acc_op, "recall": rec_op, "precision": pre_op, "f1_score": f1_score}
