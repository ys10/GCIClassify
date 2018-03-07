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
        Give a distance threshold, for all peaks within distance from mark no more than threshold.
        Pick up target peak follow these priorities
            1. nearest right peak;
            2. nearest left peak;
            3. missed.
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
    miss = list()  # missed marks
    pos_cnt = 0  # positive labeled marks count
    for mark in marks:
        left_peaks = list()
        right_peaks = list()
        """calculate a search range based on mark & threshold"""
        search_range = calculate_search_range(mark, threshold)
        """record target peaks in search range"""
        for j in range(0, len(peaks)):
            peak = peaks[j]
            if peak < search_range["left"]:
                continue
            elif peak > search_range["right"]:
                continue
            elif search_range["left"] <= peak <= mark:  # in left half search range
                left_peaks.append(j)
            elif mark < peak <= search_range["right"]:  # in right half search range
                right_peaks.append(j)
            else:
                print("mark: {}, peak: {}, threshold: {}".format(mark, peak, threshold))
                print("left_border: {}, right_border: {}".format(search_range["left"], search_range["right"]))
                raise KeyError
        """pick up the optimum peak"""
        left_peaks.sort()
        right_peaks.sort()
        if len(right_peaks) > 0:  # nearest right peak exists.
            right_peaks.sort()
            peak_idx = right_peaks[0]
        elif len(left_peaks) > 0:  # nearest right peak does not exist, but nearest left peak exists.
            left_peaks.sort()
            peak_idx = left_peaks[len(left_peaks) - 1]
        else:  # neither nearest right or left peak exists, finally miss this mark & record it.
            miss.append(mark)
            continue
        labels[peak_idx] = 1
        peak = peaks[peak_idx]
        error = abs(peak - mark)
        errors[peak_idx] = error
        pos_cnt += 1
    assert len(peaks) == len(labels) == len(errors)
    return labels, errors, miss, pos_cnt


def calculate_search_range(mark, threshold):
    search_range = {"left": mark-threshold/2, "right": mark+threshold}
    return search_range


# def label_peaks(peaks, marks, threshold):
#     """
#     Label peaks with marks.
#         Give a distance threshold, for all peaks within distance from mark no more than threshold.
#         Pick up target peak follow these priorities
#             1. nearest right peak;
#             2. nearest left peak;
#             3. missed.
#     :param peaks: peak indices.
#     :param marks: marks indices.
#     :param threshold: distance threshold between a couple of (peak, mark).
#     :return: a tuple(labels, errors, pos_cnt) where:
#         labels: peak labels.
#         errors: distance between peaks and marks(zero for negative sample)
#         miss: missed marks
#         pos_cnt: positive sample count.
#     """
#     marks.sort()
#     peaks.sort()
#     labels = [0] * len(peaks)
#     errors = [0] * len(peaks)
#     miss = list()  # missed marks
#     pos_cnt = 0  # positive labeled marks count
#     current_peak = 0  # peak index
#     for i in range(len(marks)):
#         mark = marks[i]
#         if current_peak >= len(peaks) - 1:  # finally miss this mark & record it.
#             miss.append(mark)
#             continue
#         left_peaks = []
#         right_peaks = []
#         for j in range(current_peak, len(peaks)):
#             peak = peaks[j]
#             error = abs(peak-mark)
#             if peak < mark & error <= threshold:
#                 left_peaks.append(j)
#             elif peak >= mark & error <= threshold:
#                 right_peaks.append(j)
#             elif peak > mark:  # Key step
#                 break
#         left_peaks.sort()
#         right_peaks.sort()
#         if len(right_peaks) > 0:  # nearest right peak exists.
#             right_peaks.sort()
#             peak_idx = right_peaks[0]
#         elif len(left_peaks) > 0:  # nearest right peak does not exist, but nearest left peak exists.
#             left_peaks.sort()
#             peak_idx = left_peaks[len(left_peaks) - 1]
#         else:  # neither nearest right or left peak exists, finally miss this mark & record it.
#             miss.append(mark)
#             # rate = 16000
#             # print("\tmissed mark: " + str(mark / rate))
#             # print("\tcurrent peak: " + str(peaks[current_peak] / rate))
#             continue
#         labels[peak_idx] = 1
#         peak = peaks[peak_idx]
#         error = abs(peak - mark)
#         errors[peak_idx] = error
#         pos_cnt += 1
#         current_peak = peak_idx + 1
#     assert len(peaks) == len(labels) == len(errors)
#     return labels, errors, miss, pos_cnt
#
#
# def old_label_peaks(peaks, marks, threshold):
#     """
#     Label peaks with marks.
#         Give a distance threshold, for all peaks within distance from mark no more than threshold.
#         Pick up target peak follow these priorities
#             1. nearest right peak;
#             2. missed.
#     :param peaks: peak indices.
#     :param marks: marks indices.
#     :param threshold: distance threshold between a couple of (peak, mark).
#     :return: a tuple(labels, errors, pos_cnt) where:
#         labels: peak labels.
#         errors: distance between peaks and marks(zero for negative sample)
#         miss: missed marks
#         pos_cnt: positive sample count.
#     """
#     labels = [0] * len(peaks)
#     errors = [0] * len(peaks)
#     miss = list()
#     pos_cnt = 0
#     current_peak = 0
#     for i in range(len(marks)):
#         mark = marks[i]
#         if current_peak == len(peaks):  # finally miss this mark & record it.
#             miss.append(mark)
#             continue
#         for j in range(current_peak, len(peaks)):
#             peak = peaks[j]
#             error = peak-mark
#             if peak >= mark & error <= threshold:  # label this peak & jump out of the loop.
#                 labels[j] = 1
#                 errors[j] = error
#                 pos_cnt += 1
#                 current_peak = j+1
#                 break
#             if j == len(peaks)-1:  # finally miss this mark & record it.
#                 miss.append(mark)
#     assert len(peaks) == len(labels) == len(errors)
#     return labels, errors, miss, pos_cnt


def crop_wav(wav, center, radius):
    """
    Crop wav on [center - radius, center + radius + 1], and pad 0 for out of range indices.
    :param wav: wav
    :param center: crop center
    :param radius: crop radius
    :return: a slice whose length is radius*2 +1.
    """
    left_border = center - radius
    right_border = center + radius + 1
    if left_border < 0:
        zeros = np.zeros(-left_border)
        cropped_wav = np.concatenate([zeros, wav[0: right_border]])
    elif right_border > len(wav):
        zeros = np.zeros(right_border - len(wav))
        cropped_wav = np.concatenate([wav[left_border: len(wav)], zeros])
    else:
        cropped_wav = wav[left_border: right_border]
    assert len(cropped_wav) == radius * 2 + 1
    return cropped_wav
