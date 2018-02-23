# coding=utf-8
import tensorflow as tf


class ClassifyModel(object):
    def __init__(self, name="NaiveClassifier", input_size=32, label_classes=2):
        self.name = name
        self.input_size = input_size
        self.label_classes = label_classes

    def build(self, data, reuse=None, training=False):
        with tf.variable_scope(self.name, reuse=reuse):
            features = tf.cast(data["features"], tf.float32)
            labels = tf.cast(data["labels"], tf.int32)
            output = tf.layers.batch_normalization(features, training=training, name="bn_layer_0")
            output = tf.layers.dense(inputs=output, units=16, activation=tf.nn.relu, name="hidden_layer_1")
            output = tf.layers.batch_normalization(output, training=training, name="bn_layer_1")
            output = tf.layers.dense(inputs=output, units=8, activation=tf.nn.relu, name="hidden_layer_2")
            output = tf.layers.batch_normalization(output, training=training, name="bn_layer_2")
            output = tf.layers.dense(inputs=output, units=4, activation=tf.nn.relu, name="hidden_layer_3")
            output = tf.layers.batch_normalization(output, training=training, name="bn_layer_3")
            logits = tf.layers.dense(inputs=output, units=2, activation=tf.nn.softmax, name="output_layer")
            predictions = tf.argmax(logits, axis=-1)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            acc, acc_op = tf.metrics.accuracy(labels, predictions)
            rec, rec_op = tf.metrics.recall(labels, predictions)
            pre, pre_op = tf.metrics.precision(labels, predictions)
            return {"predictions": predictions, "labels": labels, "loss": loss,
                    "accuracy": acc_op, "recall": rec_op, "precision": pre_op}
