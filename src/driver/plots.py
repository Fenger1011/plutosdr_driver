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