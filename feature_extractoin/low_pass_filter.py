# coding=utf-8
import numpy as np
from scipy.signal import filter_design as fd
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt


def butter_low_pass_coefficients(cut_off, rate, order=5):
    """
    Generate butter worth low pass filter coefficients.
    :param cut_off: cut off frequency(Hz).
    :param rate: sampling rate.
    :param order: filter order.
    :return: a tuple of ba coefficients of low pass filter.
    """
    nyq = 0.5 * rate
    normal_cut_off = cut_off / nyq
    b, a = butter(order, normal_cut_off, btype='low', analog=False)
    return b, a


def butter_low_pass_filter(data, cut_off, rate, order=5):
    """
    Filter data by butter low pass filter.
    :param data: input data.
    :param cut_off: cut off frequency.
    :param rate: sampling rate.
    :param order: filter order.
    :return: filtered data.
    """
    b, a = butter_low_pass_coefficients(cut_off, rate, order=order)
    y = lfilter(b, a, data)
    return y


def cheby2_low_pass_coefficients(cut_off, rate, ripple=0.5, attenuation=60):
    """
    Generate low pass filter coefficients.
    :param rate: sampling rate of speech waveform(Hz).
    :param cut_off: cut off frequency (Hz)
    :param ripple: pass band maximum loss
    :param attenuation: stop band min attenuation
    :return: a tuple of ba coefficients of low pass filter.
    """
    nyq = 0.5 * rate  # Nyquist frequency, half of sampling frequency(Hz).
    normal_cut_off = round(cut_off / nyq, 3)
    wp = normal_cut_off - 0.01  # end of pass band, normalized frequency
    ws = normal_cut_off + 0.01  # start of the stop band, normalized frequency
    b, a = fd.iirdesign(wp, ws, ripple, attenuation, ftype='cheby2')
    return b, a


def cheby2_low_pass_filter(data, cut_off, rate, ripple=0.5, attenuation=60):
    """
    Filter data by cheby2 filter.
    :param data: input data.
    :param cut_off: cut off frequency.
    :param rate: sampling rate.
    :param ripple: pass band maximum loss
    :param attenuation: stop band min attenuation
    :return: filtered data.
    """
    b, a = cheby2_low_pass_coefficients(cut_off, rate, ripple, attenuation)
    y = lfilter(b, a, data)
    return y


def test_butter_low_pass_filter():
    # Filter requirements.
    order = 6
    rate = 16000  # sample rate, Hz
    cut_off = 700  # desired cut_off frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_low_pass_coefficients(cut_off, rate, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * rate * w / np.pi, np.abs(h), 'b')
    plt.plot(cut_off, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cut_off, color='k')
    plt.xlim(0, 0.5 * rate)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    t = 5.0  # seconds
    n = int(t * rate)  # total number of samples
    t = np.linspace(0, t, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_low_pass_filter(data, cut_off, rate, order)

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()
    # TODO
    pass


def test_cheby2_low_pass_filter():
    rate = 16000  # sampling rate of speech waveform(Hz)
    cut_off = 700  # cut off frequency (Hz)
    ripple = 0.5  # pass band maximum loss (gpass)
    attenuation = 60  # stop band min attenuation (gstop)

    b, a = cheby2_low_pass_coefficients(cut_off, rate, ripple, attenuation)
    # Plot the frequency response.
    w, h = fd.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * rate * w / np.pi, np.abs(h), 'b')
    plt.plot(cut_off, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cut_off, color='k')
    plt.xlim(0, 0.5 * rate)
    plt.title("Low pass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    t = 5.0  # seconds
    n = int(t * rate)  # total number of samples
    t = np.linspace(0, t, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

    # Filter the data, and plot both the original and filtered signals.
    y = cheby2_low_pass_filter(data, cut_off, rate, ripple, attenuation)

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()
    # TODO
    pass


if __name__ == "__main__":
    test_butter_low_pass_filter()
    test_cheby2_low_pass_filter()
