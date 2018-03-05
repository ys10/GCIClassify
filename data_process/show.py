# coding=utf-8
import os
import wave

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

from data_process.low_pass_filter import butter_low_pass_filter

key = "kdt_028"
wave_dir = "data/origin/cmu/cmu_us_ked_timit/wav/"
wave_name = key + ".wav"
wave_path = os.path.join(wave_dir, wave_name)

mark_dir = "data/origin/cmu/cmu_us_ked_timit/marks/"
mark_name = key + ".marks"
mark_path = os.path.join(mark_dir, mark_name)

with open(mark_path, "r") as marks_file:
    marks = list()
    while 1:
        lines = marks_file.readlines(1000)
        if not lines:
            break
        for line in lines:
            marks.append(float(line))
    pass

with wave.open(wave_path, 'rb') as wav_file:
    params = wav_file.getparams()
    n_channels, width, rate, n_frames = params[:4]
    str_data = wav_file.readframes(n_frames)  # 读取音频，字符串格式
    wave_data = np.fromstring(str_data, dtype=np.int16)  # 将字符串转化为int
    # wave_data = wave_data*1.0/(max(abs(wave_data)))  # wave幅值归一化
# Filter the data, and plot both the original and filtered signals.
cut_off = 700
order = 6

t = np.arange(0, n_frames)*(1.0 / rate)
y = butter_low_pass_filter(wave_data, cut_off, rate, order)

local_min_idx = argrelextrema(y, np.less)
local_min_idx = local_min_idx[0]

# threshold = -0.015 * rate
threshold = -200
local_min_idx = [idx for idx in local_min_idx if y[idx] < threshold]
x = [idx * 1.0 / rate for idx in local_min_idx]
print("Marks number: " + str(len(marks)))
print("local minimum number: " + str(len(x)))

plt.subplot(2, 1, 1)
plt.plot(t, wave_data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
for mark in marks:
    plt.axvline(mark, color='red', linestyle="--")
plt.plot(x, y[local_min_idx], 'ks')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
