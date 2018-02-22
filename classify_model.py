# coding=utf-8
import tensorflow as tf


class ClassifyModel(object):
    def __init__(self, name="NaiveClassifier", input_size=32, label_classes=2):
        self.name = name
        self.input_size = input_size
        self.label_classes = label_classes

    def build(self, data, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            features = tf.cast(data["features"], tf.float32)
            labels = tf.cast(data["labels"], tf.int32)
            # labels = tf.one_hot(data["labels"], depth=self.label_classes, dtype=tf.float32)
            output = tf.layers.dense(inputs=features, units=16, activation=tf.nn.relu, name="hidden_layer_1")
            output = tf.layers.dense(inputs=output, units=8, activation=tf.nn.relu, name="hidden_layer_2")
            logits = tf.layers.dense(inputs=output, units=2, activation=tf.nn.softmax, name="output_layer")
            modes = tf.argmax(logits, axis=-1)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            return {"output": modes, "labels": labels, "loss": loss}
