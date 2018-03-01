# coding=utf-8
import os
import numpy as np
from scipy.signal import argrelextrema
from scipy.io import wavfile


def find_local_minimum(data, threshold=None):
    """
    Find local minimum in data.
    :param data: input data.
    :param threshold: (optional) local minimum whose value is not less than threshold won't be selected.
    :return: a 1-D array.
    """
    local_min_idx = argrelextrema(data, np.less)
    local_min_idx = local_min_idx[0]
    if threshold:
        local_min_idx = [idx for idx in local_min_idx if data[idx] < threshold]
    return local_min_idx


def file_names(file_dir):
    """
    List all file names(without extension) in target directory.
    :param file_dir:
        target directory.
    :return:
        a list containing file names.
    """
    file_names_list = list()
    for _, _, files in os.walk(file_dir):
        for file in files:
            file_names_list.append(file.split(".")[0])
    return file_names_list


def read_wav_data(path):
    """
    Read wav file.
    :param path:
        wav file path.
    :return:
        sampling rate, waveform data.
    """
    rate, data = wavfile.read(path)
    return rate, data[:]


def read_marks_data(path, rate, wave_length):
    """
    Read marks file.
    :param path:
        marks file path(containing time of gci).
    :param rate:
        sampling rate.
    :param wave_length:
        wave length.
    :return:
        an list containing the index(time * rate) of gci.
    """
    marks = list()
    with open(path) as mark_file:
        while 1:
            lines = mark_file.readlines(10000)
            if not lines:
                break
            marks.extend(map(lambda l: round(float(l) * rate), lines))
    if marks[-1] >= wave_length:
        return marks[:-2]
    return marks


def label_peaks(peaks, marks, threshold):
    """
    Label peaks with marks.
    :param peaks: peak indices.
    :param marks: marks indices.
    :param threshold: distance threshold between a couple of (peak, mark).
    :return: a tuple(labels, errors, pos_cnt) where:
        labels: peak labels.
        errors: distance between peaks and marks(zero for negative sample)
        miss: missed marks
        pos_cnt: positive sample count.
    """
    labels = [0] * len(peaks)
    errors = [0] * len(peaks)
    miss = list()
    pos_cnt = 0
    current_peak = 0
    for i in range(len(marks)):
        mark = marks[i]
        if current_peak == len(peaks) - 1:  # finally miss this mark & record it.
            miss.append(mark)
            continue
        for j in range(current_peak, len(peaks)):
            peak = peaks[j]
            error = peak-mark
            if peak >= mark & error <= threshold:  # label this peak & jump out of the loop.
                labels[j] = 1
                errors[j] = error
                pos_cnt += 1
                current_peak = j+1
                break
            if j == len(peaks)-1:  # finally miss this mark & record it.
                miss.append(mark)
    assert len(peaks) == len(labels) == len(errors)
    return labels, errors, miss, pos_cnt


def crop_wav(wav, center, radius):
    """
    Crop wav on [center - radius, center + radius + 1], and pad 0 for out of range indices.
    :param wav: wav
    :param center: crop center
    :param radius: crop radius
    :return: a slice whose length is radius*2 +1.
    """
    cropped_wav = wav[center - radius: center + radius + 1]
    # TODO
    return cropped_wav


def main():
    numbers = [3, 2, 4, 1, 7]
    num, idx = find_local_minimum(numbers)
    print(idx)
    print(num)


if __name__ == "__main__":
    main()
