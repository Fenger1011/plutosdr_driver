import adi
import numpy as np

# Make sinewave
def make_tone(fs, tone_hz, n, amplitude):
    t = np.arange(n) / fs
    iq = amplitude * np.exp(1j * 2*np.pi * tone_hz * t)
    return iq.astype(np.complex64)