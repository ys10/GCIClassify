# coding=utf-8
import os
import shutil
import random
from data_process.ops import file_names


def random_keys(path, num=50):
    all_keys = file_names(path)
    training_keys = random.sample(all_keys, num)
    validation_keys = list(set(all_keys) - set(training_keys))
    return training_keys, validation_keys


def copy_files(src_path, dst_path, keys, extension, rename_prefix=""):
    for key in keys:
        src_file = os.path.join(src_path, key+extension)
        dst_file = os.path.join(dst_path, rename_prefix+key+extension)
        print("src_file: {}".format(src_file))
        print("dst_file: {}".format(dst_file))
        shutil.copy(src_file, dst_file)


def main():
    dst_path = "data/origin/cmu/mix2/"
    wav_extension = ".wav"
    marks_extension = ".marks"
    """rab data set"""
    rab_path = "data/origin/cmu/cstr_uk_rab_diphone/"
    rab_train_keys, _ = random_keys(os.path.join(rab_path, "marks"), num=400)
    rab_valid_keys = file_names(os.path.join(rab_path, "validation_marks"))
    copy_files(os.path.join(rab_path, "wav"), os.path.join(dst_path, "wav"), rab_train_keys,
               wav_extension, rename_prefix="rab_")
    copy_files(os.path.join(rab_path, "wav"), os.path.join(dst_path, "wav"), rab_valid_keys,
               wav_extension, rename_prefix="rab_")
    copy_files(os.path.join(rab_path, "marks"), os.path.join(dst_path, "marks"), rab_train_keys,
               marks_extension, rename_prefix="rab_")
    copy_files(os.path.join(rab_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               rab_valid_keys, marks_extension, rename_prefix="rab_")
    """jmk data set"""
    jmk_path = "data/origin/cmu/cmu_us_jmk_arctic/"
    jmk_train_keys, _ = random_keys(os.path.join(jmk_path, "marks"), num=300)
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
    bdl_train_keys, _ = random_keys(os.path.join(bdl_path, "marks"), num=700)
    bdl_valid_keys = file_names(os.path.join(bdl_path, "validation_marks"))
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
    slt_train_keys, _ = random_keys(os.path.join(slt_path, "marks"), num=100)
    slt_valid_keys = file_names(os.path.join(slt_path, "validation_marks"))
    copy_files(os.path.join(slt_path, "wav"), os.path.join(dst_path, "wav"), slt_train_keys,
               wav_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "wav"), os.path.join(dst_path, "wav"), slt_valid_keys,
               wav_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "marks"), os.path.join(dst_path, "marks"), slt_train_keys,
               marks_extension, rename_prefix="slt_")
    copy_files(os.path.join(slt_path, "validation_marks"), os.path.join(dst_path, "validation_marks"),
               slt_valid_keys, marks_extension, rename_prefix="slt_")


if __name__ == "__main__":
    main()
