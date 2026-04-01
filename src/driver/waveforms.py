import adi
import numpy as np

# ============ INFO ===========
# Data from the waveforms are loaded into the TX buffer, and then transmitted.



# ============ WAVEFORMS ============
# Make sinewave
def make_tone(fs, tone_hz, amplitude=2**14, n=None):
    """Generate a complex tone with optional fixed sample count n.
    
    fs : sample rate (Hz)
    tone_hz : Frequency of oscillation (Hz)
    amplitude : Waveform amplitude (2**14 is standard)
    n : Number of samples
    """
    if n is None:
        target = 32768
        samples_per_cycle = int(round(fs / tone_hz))
        num_cycles = target // samples_per_cycle
        n = samples_per_cycle * num_cycles
    else:
        n = int(n)
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")

    t = np.arange(n) / fs
    iq = amplitude * np.exp(1j * 2*np.pi * tone_hz * t)
    return iq.astype(np.complex64)


import numpy as np

def make_chirp(fs, f0, f1, n, amplitude=2**14):
    """
    Generate a complex linear chirp with fixed number of samples.

    fs : sample rate (Hz)
    f0 : start frequency (Hz)
    f1 : end frequency (Hz)
    n  : number of samples (2**14 is standard)
    amplitude : signal amplitude
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    
    t = np.arange(n) / fs
    T = n / fs                     # total duration
    k = (f1 - f0) / T             # sweep rate
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    
    print(f"Chirp duration is {T} seconds")

    iq = amplitude * np.exp(1j * phase)
    return iq.astype(np.complex64)