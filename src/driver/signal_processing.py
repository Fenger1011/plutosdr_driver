import numpy as np

def range_from_samples(rx_samples, tx_samples, sample_rate, bandwidth, chirp_duration):
    c = 3e8
    slope = bandwidth / chirp_duration

    # Dechirp / mix down
    beat = rx_samples * np.conj(tx_samples)

    # Window
    win = np.hamming(len(beat))
    beat_win = beat * win

    # FFT
    sp = np.fft.fft(beat_win)
    sp = np.fft.fftshift(sp)
    mag = np.abs(sp)

    freq = np.fft.fftfreq(len(beat), d=1/sample_rate)
    freq = np.fft.fftshift(freq)

    # Peak beat frequency
    peak_idx = np.argmax(mag)
    f_b = abs(freq[peak_idx])

    # Range
    R = c * f_b / (2 * slope)
    return R, f_b, freq, mag