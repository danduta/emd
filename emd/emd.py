import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def emd(signal, fs):
    upper_peaks, _ = find_peaks(signal)
    lower_peaks, _ = find_peaks(-signal)

    if upper_peaks.size <= 2 or lower_peaks.size <= 2:
        return signal

    time_axis = np.linspace(0, len(signal) / fs, len(signal))
    prev_imf = signal

    sd = 1

    while upper_peaks.size > 2 and lower_peaks.size > 2 and sd > 0.3:
        upper_envelope = interp1d(upper_peaks/fs, signal[upper_peaks], kind = 'cubic', fill_value = 'extrapolate')(time_axis)
        lower_envelope = interp1d(lower_peaks/fs, signal[lower_peaks], kind = 'cubic', fill_value = 'extrapolate')(time_axis)

        upper_envelope[0:5] = 0
        upper_envelope[-5:] = 0
        lower_envelope[0:5] = 0
        lower_envelope[-5:] = 0
        average_envelope = (upper_envelope + lower_envelope) / 2

        residual = average_envelope
        imf = signal - average_envelope

        sd = np.sum(np.square(imf - prev_imf))
        print(sd)

        signal = residual
        upper_peaks, _ = find_peaks(signal)
        lower_peaks, _ = find_peaks(-signal)

    return imf, time_axis
