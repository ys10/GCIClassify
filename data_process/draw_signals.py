# coding=utf-8
import os
import numpy as np
from matplotlib import pyplot as plt
from low_pass_filter import butter_low_pass_filter
from ops import find_local_minimum, label_peaks, read_wav_data, read_marks_data


def show_wav_info(rate, raw_wav, raw_egg, filtered_wav, mark_indices, missed_mark_indices,
                  positive_label_indices, negative_label_indices):
    """"""
    """draw wav signal"""
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title(label="(a)", loc="right")
    time = np.arange(0, len(raw_wav))*(1.0 / rate)  # time.
    wav_line, = ax1.plot(time, raw_wav, 'b-', label='raw_wav')  # draw raw wave data.
    for mark_idx in list(set(mark_indices) - set(missed_mark_indices)):
        ax1.axvline(mark_idx / rate, color='red', linestyle="--")  # draw identified mark locations
    # for mark_idx in missed_mark_indices:
    #     plt.axvline(mark_idx / rate, color='purple', linestyle="--")  # draw missed mark locations
    # ax1.scatter([i / rate for i in positive_label_indices], filtered_wav[positive_label_indices],
    #             color='red', label="positive_label")
    # ax1.scatter([i / rate for i in negative_label_indices], filtered_wav[negative_label_indices],
    #             color='black', label="negative_label")

    """draw filtered wav signal"""
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.set_title(label="(b)", loc="right")
    time = np.arange(0, len(raw_wav))*(1.0 / rate)  # time.
    filtered_wav_line, = ax2.plot(time, filtered_wav, 'g-', label='filtered_wav')  # draw filtered wave data.
    for mark_idx in list(set(mark_indices) - set(missed_mark_indices)):
        ax2.axvline(mark_idx / rate, color='red', linestyle="--")  # draw identified mark locations
    # for mark_idx in missed_mark_indices:
    #     plt.axvline(mark_idx / rate, color='purple', linestyle="--")  # draw missed mark locations
    # ax2.scatter([i / rate for i in positive_label_indices], filtered_wav[positive_label_indices],
    #             color='red', label="positive_label")
    # ax2.scatter([i / rate for i in negative_label_indices], filtered_wav[negative_label_indices],
    #             color='black', label="negative_label")

    """draw egg signal"""
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.set_title(label="(c)", loc="right")
    time = np.arange(0, len(raw_egg))*(1.0 / rate)  # time.
    egg_line, = ax3.plot(time, raw_egg, 'c-', label='raw_egg')  # draw filtered wave data.
    for mark_idx in list(set(mark_indices) - set(missed_mark_indices)):
        plt.axvline(mark_idx / rate, color='red', linestyle="--")  # draw identified mark locations
    # for mark_idx in missed_mark_indices:
    #     plt.axvline(mark_idx / rate, color='purple', linestyle="--")  # draw missed mark locations
    # ax3.scatter([i / rate for i in positive_label_indices],
    #             filtered_wav[positive_label_indices], color='red', label="positive_label")
    # ax3.scatter([i / rate for i in negative_label_indices],
    #             filtered_wav[negative_label_indices], color='black', label="negative_label")

    """pic"""
    plt.xlabel('Time [sec]')
    # plt.legend([wav_line, filtered_wav_line, egg_line],
    #            ["raw wav", "filtered wav", "raw egg"], loc=4)
    plt.grid()
    plt.subplots_adjust(bottom=0.07, left=0.05, right=0.99, top=0.96, wspace=0.20, hspace=0.17)
    plt.show()


def main(key, data_set_path):
    print("key: {}".format(key))

    wav_dir = os.path.join(data_set_path, "wav/")
    wav_name = key + ".wav"
    wav_path = os.path.join(wav_dir, wav_name)

    egg_dir = os.path.join(data_set_path, "lar/")
    egg_name = key + ".wav"
    egg_path = os.path.join(egg_dir, egg_name)

    marks_dir = os.path.join(data_set_path, "marks/")
    marks_name = key + ".marks"
    marks_path = os.path.join(marks_dir, marks_name)

    """read raw wav & marks data."""
    rate, raw_wav = read_wav_data(wav_path)
    raw_wav = raw_wav.astype(np.int64)
    rate, raw_egg = read_wav_data(egg_path)
    raw_egg = raw_egg.astype(np.int64)
    wav_length = len(raw_wav)
    mark_indices = read_marks_data(marks_path, rate, wav_length)

    """filter wave by a low pass filter with 700 Hz cutoff frequency"""
    filtered_wav = butter_low_pass_filter(raw_wav, cut_off=700, rate=rate, order=6)
    length = len(raw_wav)
    delay = int(rate * 0.001)  # default low pass filter delay
    raw_wav = raw_wav[:length - delay]
    # peak_indices = find_local_minimum(filtered_wav, threshold=-200)
    # print("no-delay indices: " + str(peak_indices))
    filtered_wav = filtered_wav[delay:]
    peak_indices = find_local_minimum(filtered_wav, threshold=-200)
    # print("delay indices: " + str(peak_indices))
    print("Marks number: " + str(len(mark_indices)))
    print("local minimum number: " + str(len(peak_indices)))

    """make labels"""
    peak_mark_threshold = 0.005
    labels, errors, missed_mark_indices, pos_cnt = label_peaks(peak_indices, mark_indices,
                                                               int(peak_mark_threshold * rate))
    positive_label_indices = [peak_indices[i] for i in [idx for idx, label in enumerate(labels) if label == 1]]
    negative_label_indices = [peak_indices[i] for i in [idx for idx, label in enumerate(labels) if label == 0]]
    print("positive_label_count: {}".format(len(positive_label_indices)))
    print("negative_label_count: {}".format(len(negative_label_indices)))

    """show wave info"""
    show_wav_info(rate=rate,
                  raw_wav=raw_wav,
                  raw_egg=raw_egg,
                  filtered_wav=filtered_wav,
                  mark_indices=mark_indices,
                  missed_mark_indices=missed_mark_indices,
                  positive_label_indices=positive_label_indices,
                  negative_label_indices=negative_label_indices)


if __name__ == "__main__":
    # _key = "edx_1384"
    # _data_set_path = "data/origin/cmu/cstr_uk_rab_diphone/"
    _key = "arctic_a0001"
    _data_set_path = "data/origin/cmu/cmu_us_jmk_arctic/"
    main(_key, _data_set_path)
