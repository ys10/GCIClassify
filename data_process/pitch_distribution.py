# coding=utf-8
from matplotlib import pyplot as plt
import os
from ops import file_names


def read_marks_data(path):
    """
    Read marks file.
    :param path:
        marks file path(containing time of gci).
    :return:
        an list containing the time locations of gci.
    """
    marks = list()
    with open(path) as mark_file:
        while 1:
            lines = mark_file.readlines(10000)
            if not lines:
                break
            marks.extend(map(lambda l: float(l), lines))
    return marks


def get_pitch_period(marks_dir, threshold=0.015):
    pitch_period = list()
    print("Statistic {} pitch distribution.".format(marks_dir))
    keys = file_names(marks_dir)
    print("file number: {}".format(len(keys)))
    for key in keys:
        # print("key: {}".format(key))
        marks_name = key + ".marks"
        marks_path = os.path.join(marks_dir, marks_name)
        marks = read_marks_data(marks_path)
        diff = difference(marks)
        reduced = reduce_period(diff, threshold=threshold)
        pitch_period.extend(reduced)
    return pitch_period


def difference(seq, order=1):
    diff = list()
    length = len(seq)
    for i in range(order, length):
        diff.append(seq[i] - seq[i-order])
    return diff


def reduce_period(seq, threshold=0.015):
    seq.sort()
    reduced = list()
    for e in seq:
        if e < threshold:
            reduced.append(e)
        else:
            break
    return reduced


def main():
    rab_marks_dir = "data/origin/cmu/cstr_uk_rab_diphone/marks/"
    rab_pitch_period = get_pitch_period(rab_marks_dir, threshold=0.02)
    ked_marks_dir = "data/origin/cmu/cmu_us_ked_timit/marks/"
    ked_pitch_period = get_pitch_period(ked_marks_dir, threshold=0.02)
    jmk_marks_dir = "data/origin/cmu/cmu_us_jmk_arctic/marks/"
    jmk_pitch_period = get_pitch_period(jmk_marks_dir, threshold=0.02)
    bdl_marks_dir = "data/origin/cmu/cmu_us_bdl_arctic/marks/"
    bdl_pitch_period = get_pitch_period(bdl_marks_dir, threshold=0.02)
    slt_marks_dir = "data/origin/cmu/cmu_us_slt_arctic/marks/"
    slt_pitch_period = get_pitch_period(slt_marks_dir, threshold=0.02)
    mix2_marks_dir = "data/origin/cmu/mix2/marks/"
    mix2_pitch_period = get_pitch_period(mix2_marks_dir, threshold=0.02)
    mix3_marks_dir = "data/origin/cmu/mix3/marks/"
    mix3_pitch_period = get_pitch_period(mix3_marks_dir, threshold=0.02)
    num_bins = 200
    plt.hist(rab_pitch_period, num_bins, normed=1, facecolor='green', alpha=0.5, label="rab")
    plt.hist(ked_pitch_period, num_bins, normed=1, facecolor='blue', alpha=0.5, label="ked")
    plt.hist(jmk_pitch_period, num_bins, normed=1, facecolor='red', alpha=0.5, label="jmk")
    plt.hist(bdl_pitch_period, num_bins, normed=1, facecolor='yellow', alpha=0.5, label="bdl")
    plt.hist(slt_pitch_period, num_bins, normed=1, facecolor='purple', alpha=0.5, label="slt")
    rab_pitch_period.extend(jmk_pitch_period)
    plt.hist(rab_pitch_period, num_bins, normed=1, facecolor='pink', alpha=0.5, label="rab_jmk")
    plt.hist(mix2_pitch_period, num_bins, normed=1, facecolor='blue', alpha=0.5, label="mix2")
    plt.hist(mix3_pitch_period, num_bins, normed=1, facecolor='green', alpha=0.5, label="mix3")
    plt.title("Pitch period distribution")
    plt.legend()
    plt.show()
    pass


if __name__ == "__main__":
    main()
    pass
