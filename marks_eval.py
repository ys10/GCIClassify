# coding=utf-8
import os
from numpy import std
from matplotlib import pyplot as plt
from data_process.ops import file_names


def load_marks(marks_file_path):
    marks = list()
    with open(marks_file_path) as file:
        while 1:
            lines = file.readlines(1000)
            if not lines:
                break
            for line in lines:
                marks.append(float(line))
    return marks


def accuracy(errors, mean=0.0, radius=0.00025):
    _errors = errors
    map(lambda x: x + mean, _errors)
    map(abs, _errors)
    _errors.sort()
    total_count = len(errors)
    acc_count = 0
    for e in _errors:
        if e < radius:
            acc_count += 1
    acc_rate = acc_count / total_count
    return acc_rate


def _evaluate(est, ref, radius=0.0007):
    """
    :param est: estimated marks, list of float
    :param ref: reference marks, list of float
    :return:
    """
    missed = list()
    false = list()
    correct = list()
    errors = list()
    for ref_mark in ref:
        candidates = list()
        for est_mark in est:
            if ref_mark - radius/2 > est_mark:
                continue
            elif est_mark > ref_mark + radius:
                break
            else:
                candidates.append(est_mark)
        """label gci by correct/missed/false class"""
        if len(candidates) == 0:
            missed.append(ref_mark)
        elif len(candidates) == 1:
            correct.append(ref_mark)
            errors.append(ref_mark-candidates[0])
        else:
            false.append(ref_mark)
    result_dict = {"correct": correct, "missed": missed, "false": false,
                   "errors": errors, "total": len(ref)}
    return result_dict


def evaluate(est_path, ref_path, radius=0.0007):
    est = load_marks(est_path)
    ref = load_marks(ref_path)
    result_dict = _evaluate(est, ref, radius=radius)
    total_count = result_dict["total"]
    correct_count = len(result_dict["correct"])
    missed_count = len(result_dict["missed"])
    false_count = len(result_dict["false"])
    correct_rate = correct_count / total_count
    missed_rate = missed_count / total_count
    false_rate = false_count / total_count
    std_value = std(result_dict["errors"])
    print("correct_rate: {}, missed_rate: {}, false_rate: {}, std: {}."
          .format(correct_rate, missed_rate, false_rate, std_value))
    return {"total_count": total_count, "correct_count": correct_count,
            "missed_count": missed_count, "false_count": false_count,
            "errors": result_dict["errors"]}


def main():
    est_dir = "data/test/marks/"
    ref_dir = "data/origin/cmu/cmu_us_bdl_arctic/marks/"
    marks_extension = ".marks"
    keys = set(file_names(ref_dir)) & set(file_names(est_dir))
    total_count = 0
    correct_count = 0
    missed_count = 0
    false_count = 0
    errors = list()
    for key in keys:
        est_path = os.path.join(est_dir, key+marks_extension)
        ref_path = os.path.join(ref_dir, key+marks_extension)
        result_dict = evaluate(est_path, ref_path, radius=0.002)
        total_count += result_dict["total_count"]
        correct_count += result_dict["correct_count"]
        missed_count += result_dict["missed_count"]
        false_count += result_dict["false_count"]
        errors.extend(result_dict["errors"])
    correct_rate = correct_count / total_count
    missed_rate = missed_count / total_count
    false_rate = false_count / total_count
    std_value = std(errors)
    acc_rate = accuracy(errors, mean=-0.0005)
    print("correct_rate: {}, missed_rate: {}, false_rate: {}, std: {}, A25: {}."
          .format(correct_rate, missed_rate, false_rate, std_value, acc_rate))


def show():
    est_path = "data/test/marks/arctic_a0001.marks"
    ref_path = "data/origin/cmu/cmu_us_bdl_arctic/marks/arctic_a0001.marks"
    result_dict = evaluate(est_path, ref_path, radius=0.002)
    plt.hist(result_dict["errors"], 100, facecolor='green', alpha=0.5, label="errors")
    plt.title("Errors distribution")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
