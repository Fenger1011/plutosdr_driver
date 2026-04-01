import time
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from driver.hardware_setup import *
from driver.waveforms import *
from driver.plots import *


# ======== PLUTO PARAMETERS =========
uri = "ip:192.168.2.1"
sample_rate = 1_000_000
ts = 1 / sample_rate
rx_buffer_size = 2**18 #32768
rx_bandwidth = sample_rate
tx_bandwidth = sample_rate
rx_lo = 437e6
tx_lo = 437e6
rx_gain_mode = "manual"
rx_gain = 10
rx_channel = 0
tx_channel = 0
cyclic_tx = True
tx_attenuation = -10


# ======== SETUP HARDWARE ==========
sdr = create_pluto(uri=uri, sample_rate=sample_rate)

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

# Create chirp samples
iq = make_chirp(fs=sample_rate, f0=-100e3, f1=100e3, n=DEFAULT_TX_BUFFER_SIZE)

sdr.tx_destroy_buffer()  # Clear buffer
sdr.tx(iq)  # Send waveform to buffer


# ============ DEBUGGING INFO ===========
print(
    f"TX | ch={sdr.tx_enabled_channels}, LO={sdr.tx_lo}, BW={sdr.tx_rf_bandwidth}, "
    f"cyclic={sdr.tx_cyclic_buffer}, buf={sdr.tx_buffer_size}, "
    f"gain={sdr.tx_hardwaregain_chan0}"
)

print(
    f"RX | ch={sdr.rx_enabled_channels}, LO={sdr.rx_lo}, BW={sdr.rx_rf_bandwidth}, "
    f"buf={sdr.rx_buffer_size}, "
    f"gain={sdr.rx_hardwaregain_chan0}, mode={sdr.gain_control_mode_chan0}"
)


# ============ MAIN LOOP =============
def main():
    print("Transmitting... press Ctrl+C to stop")
    try:
        while True:
            time.sleep(10)
    finally:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()