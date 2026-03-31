import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from driver.hardware_setup import (
    create_pluto,
    configure_rx,
    receive_samples,
    DEFAULT_SAMPLE_RATE,
)

EPS = 1e-20

# Set this to True if receive_samples() returns raw Pluto ADC-style integer counts
# (for example around +/-2048 for a 12-bit converter).
# Set to False if your samples are already normalized complex floats near full scale.
NORMALIZE_TO_FULL_SCALE = True

# Approximate complex full-scale for Pluto raw sample values.
# Adjust if your driver uses a different scaling.
ADC_FULL_SCALE = 2048.0

# Exponential averaging factor.
# Smaller = smoother/slower, larger = faster/noisier.
AVG_ALPHA = 0.12


def normalize_iq_to_fs(samples: np.ndarray) -> np.ndarray:
    """
    Convert IQ samples to complex float and optionally normalize to full scale.
    """
    x = np.asarray(samples).astype(np.complex64)

    if NORMALIZE_TO_FULL_SCALE:
        x = x / ADC_FULL_SCALE

    return x


def compute_freq_axis_hz(n: int, sample_rate: float, center_freq_hz: float) -> np.ndarray:
    """
    Frequency axis in Hz for an FFT-shifted spectrum.
    """
    baseband = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    return baseband + center_freq_hz


def compute_psd_dbfs(samples: np.ndarray, window: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Compute two-sided PSD in dBFS/Hz for complex IQ samples.

    This is much better for showing noise than "power per bin", because the
    displayed level does not depend strongly on FFT size in the same way.

    If samples are normalized to full-scale complex input, then the result is dBFS/Hz.
    """
    x = normalize_iq_to_fs(samples)

    xw = x * window
    X = np.fft.fftshift(np.fft.fft(xw))

    # Window power normalization for PSD
    window_power = np.sum(window ** 2)

    # Two-sided PSD estimate for complex IQ
    psd = (np.abs(X) ** 2) / (window_power * sample_rate)

    return 10.0 * np.log10(psd + EPS)


def build_window(n: int) -> np.ndarray:
    """
    Hann window.
    """
    return np.hanning(n).astype(np.float32)


def prepare_plot():
    """
    Create the matplotlib figure and axes.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(20, 10))

    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_resizable(True, True)
    except Exception:
        pass

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dBFS/Hz)")
    ax.set_title("Pluto SDR Spectrum")
    ax.grid(True)

    plt.show(block=False)
    return fig, ax


def main():
    sdr = create_pluto(sample_rate=5e6)
    configure_rx(sdr=sdr, center_freq=99.75e6, bandwidth=250e3)

    fig, ax = prepare_plot()

    # First acquisition to initialize sizes
    samples = receive_samples(sdr)
    if samples is None or len(samples) == 0:
        raise RuntimeError("receive_samples() returned no data")

    n = len(samples)
    window = build_window(n)
    freqs_hz = compute_freq_axis_hz(n, DEFAULT_SAMPLE_RATE, sdr.rx_lo)
    freqs_mhz = freqs_hz / 1e6

    psd_dbfs = compute_psd_dbfs(samples, window, DEFAULT_SAMPLE_RATE)
    avg_psd_dbfs = psd_dbfs.copy()

    line, = ax.plot(freqs_mhz, avg_psd_dbfs, lw=1.0)

    ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax.set_ylim(-180, -20)

    try:
        while plt.fignum_exists(fig.number):
            samples = receive_samples(sdr)
            if samples is None or len(samples) == 0:
                plt.pause(0.01)
                continue

            # Rebuild dependent objects if block size changes
            if len(samples) != n:
                n = len(samples)
                window = build_window(n)
                freqs_hz = compute_freq_axis_hz(n, DEFAULT_SAMPLE_RATE, sdr.rx_lo)
                freqs_mhz = freqs_hz / 1e6

                line.set_xdata(freqs_mhz)
                ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])

                avg_psd_dbfs = None

            psd_dbfs = compute_psd_dbfs(samples, window, DEFAULT_SAMPLE_RATE)

            if avg_psd_dbfs is None:
                avg_psd_dbfs = psd_dbfs
            else:
                avg_psd_dbfs = (1.0 - AVG_ALPHA) * avg_psd_dbfs + AVG_ALPHA * psd_dbfs

            line.set_ydata(avg_psd_dbfs)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()