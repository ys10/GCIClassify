# coding=utf-8
import numpy as np
import tensorflow as tf


def get_training_set(buffer=1024, epochs=1, batch_size=16):
    training_data = _load_training_data()
    tf.logging.debug("training feature shape: " + str(training_data["features"].shape))
    tf.logging.debug("training label shape: " + str(training_data["labels"].shape))
    training_set_size = training_data["features"].shape[0]
    tf.logging.info("training set size: " + str(training_set_size))
    training_set = tf.data.Dataset.from_tensor_slices(training_data)
    training_set = training_set.shuffle(buffer)
    training_set = training_set.repeat(epochs)
    training_set = training_set.batch(batch_size)
    return training_set, training_set_size


def _load_training_data():
    x = np.load('data/artificial_extraction/uwb/X_train_p3.npy')
    y = np.load('data/artificial_extraction/uwb/y_train.npy')
    return {"features": x, "labels": y}


def get_validation_set(buffer=1024, epochs=1, batch_size=None):
    validation_data = _load_validation_data()
    tf.logging.debug("validation feature shape: " + str(validation_data["features"].shape))
    tf.logging.debug("validation label shape: " + str(validation_data["labels"].shape))
    validation_set_size = validation_data["features"].shape[0]
    tf.logging.info("validation set size: " + str(validation_set_size))
    validation_set = tf.data.Dataset.from_tensor_slices(validation_data)
    validation_set = validation_set.shuffle(buffer)
    validation_set = validation_set.repeat(epochs)
    if not batch_size:
        validation_set = validation_set.batch(validation_set_size)
    else:
        validation_set = validation_set.batch(batch_size)
    return validation_set, validation_set_size


def _load_validation_data():
    x = np.load('data/artificial_extraction/uwb/X_test_p3.npy')
    y = np.load('data/artificial_extraction/uwb/y_test.npy')
    return {"features": x, "labels": y}
