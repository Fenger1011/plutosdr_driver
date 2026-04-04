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


def plot_s11(filepath, label=None, figsize=(10, 5)):
    freq_hz = []
    re = []
    im = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 3:
                freq_hz.append(float(parts[0]))
                re.append(float(parts[1]))
                im.append(float(parts[2]))

    freq_hz = np.array(freq_hz)
    s11 = np.array(re) + 1j * np.array(im)

    # Magnitude in dB
    mag_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-15))

    freq_ghz = freq_hz / 1e9

    if label is None:
        label = filepath.split("/")[-1]

    # Single plot (fixed)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(freq_ghz, mag_db, linewidth=2)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S₁₁| (dB)")
    ax.set_title(f"Log Periodic {label}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return freq_hz, s11


