# coding=utf-8
import os
import tqdm
import argparse
import tensorflow as tf
import numpy as np
from feature_extraction.extract_model import GenerateModel
from model_loader import load_model
from data_process.low_pass_filter import butter_low_pass_filter
from data_process.ops import file_names, read_wav_data, find_local_minimum, crop_wav

marks_extension = ".marks"
wav_extension = ".wav"
cut_off = 700  # cut off frequency (Hz)
filter_order = 6
crop_radius = 240  # number of sample
# local_minimum_threshold = -0.015  # normalized amplitude threshold
local_minimum_threshold = -200  # amplitude threshold
peak_mark_threshold = 0.005  # (second) threshold used in labeling peak indices with marks data


def get_args():
    parser = argparse.ArgumentParser(description="GenerateMarks")
    parser.add_argument("--save_path", type=str, default="./save/bdl/")
    parser.add_argument("--log_path", type=str, default="./log/")
    parser.add_argument("--wav_src_dir", type=str, default="./data/test/wav/")
    parser.add_argument("--marks_dst_dir", type=str, default="./data/test/marks/")
    return parser.parse_args()


def package_inputs(raw_wav, peak_indices, radius):
    inputs = list()
    for i in range(len(peak_indices)):
        peak_idx = peak_indices[i]
        cropped_wav = crop_wav(raw_wav, peak_idx, radius)
        inputs.append(cropped_wav)
        # inputs.append(map(str, cropped_wav))
    return inputs


def package_input_features(wav_src_path):
    """"""
    """read raw wav & marks data."""
    rate, raw_wav = read_wav_data(wav_src_path)
    raw_wav = raw_wav.astype(np.int64)
    """filter raw wav"""
    filtered_wav = butter_low_pass_filter(raw_wav, cut_off=cut_off, rate=rate, order=filter_order)
    """delay filtered wav"""
    length = len(raw_wav)
    delay = int(rate * 0.001)  # default low pass filter delay
    raw_wav = raw_wav[:length - delay]
    filtered_wav = filtered_wav[delay:]
    """find negative peaks in filtered wav"""
    peak_indices = find_local_minimum(filtered_wav, threshold=local_minimum_threshold)
    """package input data"""
    wav = package_inputs(raw_wav, peak_indices, crop_radius)
    input_features = {"wav": wav, "rate": rate, "peak_indices": peak_indices}
    return input_features


def generate_marks(peak_indices, peak_labels, rate):
    marks = list()
    for i in range(len(peak_labels)):
        if peak_labels[i] == 1:
            marks.append(str(round(peak_indices[i] / rate, 5)) + "\n")
    return marks


def save_marks(marks_dst_path, marks):
    with open(marks_dst_path, "w") as f:
        f.writelines(marks)
    pass


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = get_args()
    tf.logging.info("Load wav from {}".format(args.wav_src_dir))
    tf.logging.info("Generate marks to {}".format(args.marks_dst_dir))
    net = GenerateModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            inputs = tf.placeholder(tf.float32, shape=(None, crop_radius*2 + 1))

        with tf.variable_scope("extract_model"):
            tensor_dict = net.build(inputs, training=False)
            predictions = tensor_dict["predictions"]
            saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        save_path = os.path.join(args.save_path, net.name)
        if not load_model(saver, sess, save_path):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        global_step_eval = 0
        keys = file_names(args.wav_src_dir)
        testing_steps = len(keys)
        pbar = tqdm.tqdm(total=testing_steps)
        pbar.update(global_step_eval)
        for key in keys:
            tf.logging.debug("File names: {}".format(key))
            input_features = package_input_features(os.path.join(args.wav_src_dir, key + wav_extension))
            predictions_eval = sess.run(predictions, feed_dict={inputs: input_features["wav"]})
            marks = generate_marks(input_features["peak_indices"], predictions_eval, input_features["rate"])
            tf.logging.debug("Number of marks: {}".format(len(marks)))
            save_marks(os.path.join(args.marks_dst_dir, key + marks_extension), marks)
            global_step_eval += 1
            pbar.update(1)
    tf.logging.info("Congratulations!")


if __name__ == '__main__':
    main()
