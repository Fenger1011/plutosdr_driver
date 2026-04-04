import matplotlib
matplotlib.use("TkAgg")
from matplotlib.ticker import FuncFormatter

import time
import numpy as np

from driver.hardware_setup import *
from driver.waveforms import *
from driver.plots import *
from driver.signal_processing import *


# ======== PLUTO PARAMETERS =========
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
rx_buffer_size = int(np.ceil(N_fft * fs / fs_out))  # Sets RX buffer to correct
print(rx_buffer_size)



tx_attenuation = -10







# ======== SETUP HARDWARE ==========
sdr = create_pluto(uri=uri, fs=fs)

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

# ============ CREATE SIGNAL & TX ============
iq_tx = make_tone(fs=fs, tone_hz=1e6, amplitude=2**1)
#iq_tx = make_chirp(fs=fs, f0=-2e6, f1=2e6, n=2**10)

sdr.tx_destroy_buffer()
sdr.tx(iq_tx)

print_pluto_config(sdr)

# ============ MAIN LOOP =============
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