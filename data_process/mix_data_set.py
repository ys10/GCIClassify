# coding=utf-8
import os
import shutil
import random
from data_process.ops import file_names


def random_keys(path, num=50):
    all_keys = file_names(path)
    training_keys = random.sample(all_keys, num)
    testing_keys = list(set(all_keys) - set(training_keys))
    return training_keys, testing_keys


def all_keys(path):
    keys = file_names(path)
    return keys


def copy_files(src_path, dst_path, keys, extension, rename_prefix=""):
    for key in keys:
        print(key)
        print("src_file: {}".format(src_path+key))
        print("dst_file: {}".format(dst_path+key))
        src_file = os.path.join(src_path, key+extension)
        dst_file = os.path.join(dst_path, rename_prefix+key+extension)
        shutil.copy(src_file, dst_file)


def make_mix2_data_set():
    dst_path = "data/origin/cmu/mix2/"
    wav_extension = ".wav"
    marks_extension = ".marks"
    """jmk data set"""
    jmk_path = "data/origin/cmu/cmu_us_jmk_arctic/"
    jmk_train_keys, _ = random_keys(os.path.join(jmk_path, "marks"), num=400)
    jmk_valid_keys = file_names(os.path.join(jmk_path, "validation_marks"))
    copy_files(os.path.join(jmk_path, "wav"), os.path.join(dst_path, "wav"), jmk_train_keys,
               wav_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "wav"), os.path.join(dst_path, "wav"), jmk_valid_keys,
               wav_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "marks"), os.path.join(dst_path, "marks"), jmk_train_keys,
               marks_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               jmk_valid_keys, marks_extension, rename_prefix="jmk_")
    """bdl data set"""
    bdl_path = "data/origin/cmu/cmu_us_bdl_arctic/"
    bdl_train_keys, _ = random_keys(os.path.join(bdl_path, "marks"), num=800)
    bdl_valid_keys, _ = random_keys(os.path.join(bdl_path, "validation_marks"), num=30)
    copy_files(os.path.join(bdl_path, "wav"), os.path.join(dst_path, "wav"), bdl_train_keys,
               wav_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "wav"), os.path.join(dst_path, "wav"), bdl_valid_keys,
               wav_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "marks"), os.path.join(dst_path, "marks"), bdl_train_keys,
               marks_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               bdl_valid_keys, marks_extension, rename_prefix="bdl_")


def make_mix4_data_set():
    dst_path = "data/origin/cmu/mix4/"
    wav_extension = ".wav"
    marks_extension = ".marks"
    """jmk data set"""
    jmk_path = "data/origin/cmu/cmu_us_jmk_arctic/"
    jmk_train_keys, _ = random_keys(os.path.join(jmk_path, "marks"), num=400)
    jmk_valid_keys, _ = random_keys(os.path.join(jmk_path, "validation_marks"), num=30)
    copy_files(os.path.join(jmk_path, "wav"), os.path.join(dst_path, "wav"), jmk_train_keys,
               wav_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "wav"), os.path.join(dst_path, "wav"), jmk_valid_keys,
               wav_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "marks"), os.path.join(dst_path, "marks"), jmk_train_keys,
               marks_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               jmk_valid_keys, marks_extension, rename_prefix="jmk_")
    """bdl data set"""
    bdl_path = "data/origin/cmu/cmu_us_bdl_arctic/"
    bdl_train_keys, _ = random_keys(os.path.join(bdl_path, "marks"), num=800)
    bdl_valid_keys, _ = random_keys(os.path.join(bdl_path, "validation_marks"), num=30)
    copy_files(os.path.join(bdl_path, "wav"), os.path.join(dst_path, "wav"), bdl_train_keys,
               wav_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "wav"), os.path.join(dst_path, "wav"), bdl_valid_keys,
               wav_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "marks"), os.path.join(dst_path, "marks"), bdl_train_keys,
               marks_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               bdl_valid_keys, marks_extension, rename_prefix="bdl_")
    """slt data set"""
    slt_path = "data/origin/cmu/cmu_us_slt_arctic/"
    slt_train_keys, _ = random_keys(os.path.join(slt_path, "marks"), num=400)
    slt_valid_keys, _ = random_keys(os.path.join(slt_path, "validation_marks"), num=30)
    copy_files(os.path.join(slt_path, "wav"), os.path.join(dst_path, "wav"), slt_train_keys,
               wav_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "wav"), os.path.join(dst_path, "wav"), slt_valid_keys,
               wav_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "marks"), os.path.join(dst_path, "marks"), slt_train_keys,
               marks_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               slt_valid_keys, marks_extension, rename_prefix="slt_")
    """ked data set"""
    ked_path = "data/origin/cmu/cmu_us_ked_timit/"
    ked_train_keys, _ = random_keys(os.path.join(ked_path, "marks"), num=400)
    ked_valid_keys, _ = random_keys(os.path.join(ked_path, "validation_marks"), num=30)
    copy_files(os.path.join(ked_path, "wav"), os.path.join(dst_path, "wav"), ked_train_keys,
               wav_extension, rename_prefix="ked_")
    copy_files(os.path.join(ked_path, "wav"), os.path.join(dst_path, "wav"), ked_valid_keys,
               wav_extension, rename_prefix="ked_")
    copy_files(os.path.join(ked_path, "marks"), os.path.join(dst_path, "marks"), ked_train_keys,
               marks_extension, rename_prefix="ked_")
    copy_files(os.path.join(ked_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               ked_valid_keys, marks_extension, rename_prefix="ked_")


def make_mix4_test_data_set():
    dst_path = "data/origin/cmu/mix4/"
    marks_extension = ".marks"
    """jmk data set"""
    jmk_path = "data/origin/cmu/cmu_us_jmk_arctic/"
    jmk_train_keys, jmk_test_keys = random_keys(os.path.join(jmk_path, "marks"), num=400)
    jmk_valid_keys, _ =random_keys(os.path.join(jmk_path, "validation_marks"), num=50)
    copy_files(os.path.join(jmk_path, "marks"), os.path.join(dst_path, "marks"),
               jmk_train_keys, marks_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               jmk_valid_keys, marks_extension, rename_prefix="jmk_")
    copy_files(os.path.join(jmk_path, "marks"), os.path.join(dst_path, "jmk_testing_marks"),
               jmk_test_keys, marks_extension, rename_prefix="jmk_")
    """bdl data set"""
    bdl_path = "data/origin/cmu/cmu_us_bdl_arctic/"
    bdl_train_keys, bdl_test_keys = random_keys(os.path.join(bdl_path, "marks"), num=800)
    bdl_valid_keys = random_keys(os.path.join(bdl_path, "validation_marks"), num=50)
    copy_files(os.path.join(bdl_path, "marks"), os.path.join(dst_path, "marks"),
               bdl_train_keys, marks_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               bdl_valid_keys, marks_extension, rename_prefix="bdl_")
    copy_files(os.path.join(bdl_path, "marks"), os.path.join(dst_path, "bdl_testing_marks"),
               bdl_test_keys, marks_extension, rename_prefix="bdl_")
    """slt data set"""
    slt_path = "data/origin/cmu/cmu_us_slt_arctic/"
    slt_train_keys, slt_test_keys = random_keys(os.path.join(slt_path, "marks"), num=400)
    slt_valid_keys, _ = random_keys(os.path.join(slt_path, "validation_marks"), num=50)
    copy_files(os.path.join(slt_path, "marks"), os.path.join(dst_path, "marks"), slt_train_keys,
               marks_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               slt_valid_keys, marks_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "marks"), os.path.join(dst_path, "slt_testing_marks"),
               slt_test_keys, marks_extension, rename_prefix="slt_")
    """ked data set"""
    ked_path = "data/origin/cmu/cmu_us_ked_timit/"
    ked_train_keys, _ = random_keys(os.path.join(ked_path, "marks"), num=400)
    ked_valid_keys = file_names(os.path.join(ked_path, "validation_marks"))
    ked_test_keys = list(set(all_keys(os.path.join(ked_path, "marks")))-set(ked_train_keys))
    copy_files(os.path.join(ked_path, "marks"), os.path.join(dst_path, "marks"), ked_train_keys,
               marks_extension, rename_prefix="ked_")
    copy_files(os.path.join(ked_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               ked_valid_keys, marks_extension, rename_prefix="ked_")
    copy_files(os.path.join(ked_path, "marks"), os.path.join(dst_path, "ked_testing_marks"),
               ked_test_keys, marks_extension, rename_prefix="ked_")


if __name__ == "__main__":
    make_mix4_data_set()
