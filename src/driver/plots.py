import matplotlib.pyplot as plt
import numpy as np

def plot_iq_time(iq, fs, num_samples=2000):
    t = np.arange(len(iq)) / fs

    plt.figure()
    plt.plot(t[:num_samples], iq.real[:num_samples], label="I (real)")
    plt.plot(t[:num_samples], iq.imag[:num_samples], label="Q (imag)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("IQ vs Time")
    plt.legend()
    plt.grid()

    plt.show()


def compute_fft_dbfs(rx_samples, ts):
    num_samples = len(rx_samples)

    win = np.hamming(num_samples)
    y = rx_samples * win

    sp = np.abs(np.fft.fft(y))
    sp = sp[1:-1]
    sp = np.fft.fftshift(sp)

    s_mag = sp / (np.sum(win) / 2)
    s_dbfs = 20 * np.log10(np.maximum(s_mag / (2**12), 1e-15))

    xf = np.fft.fftfreq(num_samples, ts)
    xf = np.fft.fftshift(xf[1:-1]) / 1e6

    return xf, s_dbfs