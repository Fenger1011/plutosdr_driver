import os
import sys

import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib.ticker import FuncFormatter

# Ensure the src package root is on sys.path when running this file directly.
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

import time
import numpy as np

from driver.hardware_setup import *
from driver.waveforms import *
from driver.plots import *
from driver.signal_processing import *


# ====================== PLUTO PARAMETERS ======================
uri = "ip:192.168.2.1"

fs = 100_000   
ts = 1 / fs

rx_lo = 2.175e9
tx_lo = 2.175e9

rx_gain_mode = "manual"
rx_gain = 10
tx_attenuation = -10

rx_channel = 0
tx_channel = 0
cyclic_tx = True    

rx_bandwidth = fs
tx_bandwidth = fs

# We want an effective sampling rate of 16.240 kS/s
fs_out = 16_240
N_fft = 4096
rx_buffer_size = int(np.ceil(N_fft * fs / fs_out))  # Sets RX buffer to correct length for FFT size

# ====================== SETUP HARDWARE ======================
sdr = create_pluto(uri=uri, sample_rate=fs)

configure_rx(
    sdr=sdr,
    center_freq=rx_lo,
    bandwidth=rx_bandwidth,
    buffer_size=rx_buffer_size,
    gain_mode=rx_gain_mode,
    gain=rx_gain,
    channel=rx_channel,
)

configure_tx(
    sdr=sdr,
    center_freq=tx_lo,
    bandwidth=tx_bandwidth,
    cyclic=cyclic_tx,
    attenuation=tx_attenuation,
    channel=tx_channel,
)

# ====================== CREATE SIGNAL ======================
tone_freq = fs/10 # 10kHz
iq_tx = make_tone(fs=fs, tone_hz=tone_freq, amplitude=2**1)

plot_iq_time(iq_tx, fs=fs)


# ====================== TX ======================
sdr.tx_destroy_buffer()
sdr.tx(iq_tx)

print_pluto_config(sdr)

# ====================== MAIN LOOP ======================
def main():
    

    try:
        while True:
            rx_samples = sdr.rx()

            xf, s_dbfs = compute_fft_dbfs(rx_samples, ts)

            

    except KeyboardInterrupt:
        print("\nStopping...")

    except Exception as e:
        print("Error:", e)
        raise

    finally:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()