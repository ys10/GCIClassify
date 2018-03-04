# coding=utf-8
import tensorflow as tf


def parse_example(record):
    features = tf.parse_single_example(record,
                                       features={
                                           "wav": tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'error': tf.FixedLenFeature([], tf.int64),
                                       })
    wav = tf.to_float(tf.decode_raw(features["wav"], tf.int64))
    label = tf.cast(features['label'], tf.int32)
    error = tf.cast(features['error'], tf.int32)
    return {"wav": wav, "label": label, "error": error}


def get_training_set(buffer=1024, epochs=1, batch_size=16):
    path = "data/crop_sample/training.tfrecords"
    data_set = tf.data.TFRecordDataset(path)
    data_set = data_set.map(parse_example)
    data_set = data_set.shuffle(buffer)
    data_set = data_set.repeat(epochs)
    data_set = data_set.batch(batch_size)
    return data_set


def get_validation_set(buffer=1024, epochs=1, batch_size=None):
    path = "data/crop_sample/validation.tfrecords"
    data_set = tf.data.TFRecordDataset(path)
    data_set = data_set.map(parse_example)
    data_set = data_set.shuffle(buffer)
    data_set = data_set.repeat(epochs)
    data_set = data_set.batch(batch_size)
    return data_set


def get_testing_set(key="uwb", buffer=1024, epochs=1, batch_size=None):
    path = "data/crop_sample/" + key + ".tfrecords"
    data_set = tf.data.TFRecordDataset(path)
    data_set = data_set.map(parse_example)
    data_set = data_set.shuffle(buffer)
    data_set = data_set.repeat(epochs)
    data_set = data_set.batch(batch_size)
    return data_set
