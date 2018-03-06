# coding=utf-8
import os
import numpy as np
from matplotlib import pyplot as plt
from low_pass_filter import butter_low_pass_filter
from ops import find_local_minimum, label_peaks, read_wav_data, read_marks_data


def show_wav_info(rate, raw_wav, filtered_wav, mark_indices, positive_label_indices, negative_label_indices):
    plt.subplot(2, 1, 1)
    time = np.arange(0, len(raw_wav))*(1.0 / rate)  # time.
    plt.plot(time, raw_wav, 'b-', label='raw_wav')  # draw raw wave data.
    plt.plot(time, filtered_wav, 'g-', linewidth=2, label='filtered_wav')  # draw filtered wave data.
    for mark_idx in mark_indices:
        plt.axvline(mark_idx / rate, color='yellow', linestyle="--")  # draw mark locations
    plt.scatter([i / rate for i in positive_label_indices], filtered_wav[positive_label_indices],
                color='red', label="positive_label")
    plt.scatter([i / rate for i in negative_label_indices], filtered_wav[negative_label_indices],
                color='black', label="negative_label")
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.35)
    plt.show()


def main():
    key = "kdt_028"
    wav_dir = "data/origin/cmu/cmu_us_ked_timit/wav/"
    wav_name = key + ".wav"
    wav_path = os.path.join(wav_dir, wav_name)

    marks_dir = "data/origin/cmu/cmu_us_ked_timit/marks/"
    marks_name = key + ".marks"
    marks_path = os.path.join(marks_dir, marks_name)

    print("key: {}".format(key))
    """read raw wav & marks data."""
    rate, raw_wav = read_wav_data(wav_path)
    raw_wav = raw_wav.astype(np.int64)
    wav_length = len(raw_wav)
    mark_indices = read_marks_data(marks_path, rate, wav_length)

    filtered_wav = butter_low_pass_filter(raw_wav, cut_off=700, rate=rate, order=6)
    peak_indices = find_local_minimum(filtered_wav, threshold=-200)
    print("Marks number: " + str(len(mark_indices)))
    print("local minimum number: " + str(len(peak_indices)))

    """make labels"""
    peak_mark_threshold = 0.005
    labels, errors, miss, pos_cnt = label_peaks(peak_indices, mark_indices, int(peak_mark_threshold * rate))
    positive_label_indices = [peak_indices[i] for i in [idx for idx, label in enumerate(labels) if label == 1]]
    negative_label_indices = [peak_indices[i] for i in [idx for idx, label in enumerate(labels) if label == 0]]
    show_wav_info(rate, raw_wav, filtered_wav, mark_indices, positive_label_indices, negative_label_indices)


if __name__ == "__main__":
    main()
