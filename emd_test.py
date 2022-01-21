import matplotlib.pyplot as plt
from emd.emd import emd
from numpy import random, linspace, abs
from scipy.fftpack import fft

if __name__ == '__main__':
    random.seed(47)

    sample_fs = 100
    seconds = 5
    signal = random.randn(seconds * sample_fs)

    imf, time_axis = emd(signal, sample_fs)

    plt.title('Original signal vs IMF')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal value')
    plt.plot(time_axis, signal, label='Original signal')
    plt.plot(time_axis, imf, label='IMF')
    plt.legend()

    time_axis = linspace(0, 50, 512)
    signal_fft = abs(fft(signal,1024))[0:512]
    imf_fft = abs(fft(imf, 1024))[0:512]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('FFT transforms of the original signal and the IMF')
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Weight')
    ax1.plot(time_axis, signal_fft, color='#f54242')
    ax1.set_title('Original signal\'s FFT transform', fontsize=10)
    ax2.plot(time_axis, imf_fft, color='#69f542')
    ax2.set_title('IMF\'s FFT transform', fontsize=10)

    plt.show()

