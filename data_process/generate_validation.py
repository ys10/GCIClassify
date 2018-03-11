# coding=utf-8
import os
import shutil
import random
from data_process.ops import file_names

validation_marks_path = "data/origin/cmu/cmu_us_slt_arctic/validation_marks/"
marks_path = "data/origin/cmu/cmu_us_slt_arctic/marks/"
wav_path = "data/origin/cmu/cmu_us_slt_arctic/wav/"

marks_keys = file_names(marks_path)
wav_keys = file_names(wav_path)

keys = list(set(marks_keys) - set(wav_keys))
print(keys)

for key in keys:
    my_file = os.path.join(marks_path, key + ".marks")
    os.remove(my_file)

marks_keys = file_names(marks_path)

intersection_keys = list(set(marks_keys) & set(wav_keys))
print(intersection_keys)
validation_keys = random.sample(intersection_keys, 100)
for key in validation_keys:
    shutil.move(os.path.join(marks_path, key + ".marks"), os.path.join(validation_marks_path, key + ".marks"))
