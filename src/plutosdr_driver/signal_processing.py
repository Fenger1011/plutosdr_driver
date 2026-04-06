import numpy as np


def compute_fft_dbfs(rx_samples, ts):
    num_samples = len(rx_samples)

    win = np.hamming(num_samples)
    y = rx_samples * win    # Apply Hamming window

    X = np.fft.fftshift(np.fft.fft(y))  # Computes symmettric FFT
    xf = np.fft.fftshift(np.fft.fftfreq(num_samples, d=ts)) # Create freq. axis

    s_mag = np.abs(X) / (np.sum(win) / 2)   # Magnitude + scaling
    s_dbfs = 20 * np.log10(np.maximum(s_mag / (2**12), 1e-15))  # Convert magnitude of dBFS --> Signal strength relative to ADC scale

    return xf, s_dbfs   # Return freq. scale and magnitude
