from random import randint
import matplotlib.pyplot as plt
from emd.emd import emd
from numpy import random, linspace, abs
from scipy.fftpack import fft
from scipy.signal.signaltools import wiener
from time import time

if __name__ == '__main__':
    tests = 5

    for i in range(tests):
        sample_fs = 100
        seconds = 5
        signal = random.randn(seconds * sample_fs)

        t0 = time()
        imf, time_axis = emd(signal, sample_fs)
        emd_time = (time() - t0) * 1000

        t0 = time()
        signal_wiener = wiener(signal)
        wiener_time = (time() - t0) * 1000

        print(emd_time, wiener_time)

        plt.figure()
        plt.title('Original signal vs IMF')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal value')
        plt.plot(time_axis, signal, label='Original signal')
        plt.plot(time_axis, imf, label='EMD')
        plt.plot(time_axis, signal_wiener, 'y', label='Wiener', linewidth=0.5)
        plt.legend()

        plt.savefig(f'assets/comparison{i}.png')

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

        plt.savefig(f'assets/fft{i}.png')

    plt.show()

