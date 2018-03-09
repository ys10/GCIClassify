# coding=utf-8
import tensorflow as tf
import numpy as np

from low_pass_filter import butter_low_pass_filter
from ops import file_names, read_wav_data, read_marks_data, find_local_minimum, label_peaks, crop_wav

marks_path = "data/origin/cmu/cstr_uk_rab_diphone/marks/"
wav_path = "data/origin/cmu/cstr_uk_rab_diphone/wav/"
marks_extension = ".marks"
wav_extension = ".wav"
data_path = "data/crop_sample/rab_testing.tfrecords"
cut_off = 700  # cut off frequency (Hz)
filter_order = 6
crop_radius = 240  # number of sample
# local_minimum_threshold = -0.015  # normalized amplitude threshold
local_minimum_threshold = -200  # amplitude threshold
peak_mark_threshold = 0.005  # (second) threshold used in labeling peak indices with marks data


def training_data_feature(wav, label, error):
    feature_dict = {
        "wav": tf.train.Feature(bytes_list=tf.train.BytesList(value=[wav.tobytes()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "error": tf.train.Feature(int64_list=tf.train.Int64List(value=[error])),
    }
    return feature_dict


def main():
    total_marks = 0
    total_missed = 0
    total_peaks = 0
    with tf.python_io.TFRecordWriter(data_path) as writer:
        print("Process training data start!")
        keys = file_names(marks_path)
        print("file number: {}".format(len(keys)))
        for key in keys:
            print("key: {}".format(key))
            """read raw wav & marks data."""
            rate, raw_wav = read_wav_data(wav_path + key + wav_extension)
            raw_wav = raw_wav.astype(np.int64)
            # raw_wav = raw_wav*1.0/(max(abs(raw_wav)))  # wave amplitude normalization
            # print(raw_wav)
            wav_length = len(raw_wav)
            marks_data = read_marks_data(marks_path + key + marks_extension, rate, wav_length)
            """filter raw wav"""
            filtered_wav = butter_low_pass_filter(raw_wav, cut_off=cut_off, rate=rate, order=filter_order)
            """delay filtered wav"""
            length = len(raw_wav)
            delay = int(rate * 0.001)  # default low pass filter delay
            raw_wav = raw_wav[:length - delay]
            filtered_wav = filtered_wav[delay:]
            """find negative peaks in filtered wav"""
            peak_indices = find_local_minimum(filtered_wav, threshold=local_minimum_threshold)
            """make labels"""
            labels, errors, miss, pos_cnt = label_peaks(peak_indices, marks_data, int(peak_mark_threshold * rate))
            """package output data"""
            for i in range(len(peak_indices)):
                peak_idx = peak_indices[i]
                # cropped_wav = crop_wav(filtered_wav, peak_idx, crop_radius)
                cropped_wav = crop_wav(raw_wav, peak_idx, crop_radius)
                error = errors[i]
                example = tf.train.Example(features=tf.train.Features(
                    feature=training_data_feature(cropped_wav, labels[i], error)))
                writer.write(example.SerializeToString())
            print("missed: " + str([m / rate for m in miss]))
            print("utterance marks: {}, missed marks: {}, positive sample: {}, negative sample: {}"
                  .format(len(marks_data), len(miss), pos_cnt, len(peak_indices)-pos_cnt))
            total_marks += len(marks_data)
            total_missed += len(miss)
            total_peaks += len(peak_indices)
            if len(miss) / len(marks_data) > 0.01:
                print("warning!")
            # break
        print("total marks: {}".format(total_marks))
        print("total missed: {}".format(total_missed))
        print("total peaks: {}".format(total_peaks))
    print("Done!")


if __name__ == '__main__':
    main()
