# coding=utf-8
import os
import numpy as np
import tensorflow as tf

data_path = "data/artificial_extraction"


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
    x = np.load(os.path.join(data_path, "uwb", "X_train_p3.npy"))
    y = np.load(os.path.join(data_path, "uwb", "y_train.npy"))
    return {"features": x, "labels": y}


def get_validation_set(buffer=1024, epochs=1, batch_size=None):
    validation_data = _load_testing_data("uwb")
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


def get_testing_set(key="uwb", buffer=1024, epochs=1, batch_size=None):
    testing_data = _load_testing_data(key)
    tf.logging.info("testing set name: " + key)
    tf.logging.debug("testing feature shape: " + str(testing_data["features"].shape))
    tf.logging.debug("testing label shape: " + str(testing_data["labels"].shape))
    testing_set_size = testing_data["features"].shape[0]
    tf.logging.info("testing set size: " + str(testing_set_size))
    testing_set = tf.data.Dataset.from_tensor_slices(testing_data)
    testing_set = testing_set.shuffle(buffer)
    testing_set = testing_set.repeat(epochs)
    if not batch_size:
        testing_set = testing_set.batch(testing_set_size)
    else:
        testing_set = testing_set.batch(batch_size)
    return testing_set, testing_set_size


def _load_testing_data(key):
    x = np.load(os.path.join(data_path, key, "X_test_p3.npy"))
    y = np.load(os.path.join(data_path, key, "y_test.npy"))
    return {"features": x, "labels": y}
