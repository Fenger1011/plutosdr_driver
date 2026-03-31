import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from driver.hardware_setup import *

# ======== PLUTO PARAMETERS =========
uri = "ip:192.168.2.1"
sample_rate = 5e6
rx_buffer_size = 32768
tx_buffer_size = 32768
rx_bandwidth = sample_rate
tx_bandwidth = DEFAULT_SAMPLE_RATE
rx_lo = 2_400_000_000
tx_lo = 2_400_000_000
rx_gain_mode = "slow_attack"
rx_channel = 0
tx_channel = 0
cyclic_tx = False
tx_attenuation = -10


# ======== SETUP HARDWARE ==========
# Create object
sdr = create_pluto(uri=uri, sample_rate=sample_rate)

# Configure RX
configure_rx(
    sdr=sdr,
    center_freq=rx_lo,
    bandwidth=sample_rate,
    buffer_size=DEFAULT_RX_BUFFER_SIZE,
    gain_mode="slow_attack",
    channel=0
)

# Configure TX
configure_tx(
    sdr=sdr,
    center_freq=tx_lo,
    bandwidth=sample_rate,
    buffer_size=DEFAULT_TX_BUFFER_SIZE,
    cyclic=True,
    attenuation=-10,
    channel=0
)






