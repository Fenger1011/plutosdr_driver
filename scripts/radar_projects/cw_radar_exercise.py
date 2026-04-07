# Import driver
from plutosdr_driver import *


# ====================== PLUTO PARAMETERS ======================
uri = "ip:192.168.2.1"

fs =                                # Sample rate
ts =                                # Sample period

rx_lo =                             # RX LO
tx_lo =                             # TX LO

lam = 3e8 / tx_lo                   # Wavelength corresponding to TX LO

rx_gain_mode = "manual"             # Gain mode
rx_gain =                           # Gain in dB
tx_attenuation =                    # Attenuation in dB

rx_channel = 0                      # Active RX channel
tx_channel = 0                      # Active TX channel
cyclic_tx = True                    # TX cyclic buffer True

rx_bandwidth = fs                   # RX bandwidth
tx_bandwidth = fs                   # TX bandwidth

rx_buffer_size =                    # RX buffer length


# ====================== SETUP HARDWARE ======================
# Create SDR object
sdr = create_pluto(uri=uri, sample_rate=fs)

# Configure RX
configure_rx(
    sdr=sdr,
    center_freq=rx_lo,
    bandwidth=rx_bandwidth,
    buffer_size=rx_buffer_size,
    gain_mode=rx_gain_mode,
    gain=rx_gain,
    channel=rx_channel,
)

# Configure TX
configure_tx(
    sdr=sdr,
    center_freq=tx_lo,
    bandwidth=tx_bandwidth,
    cyclic=cyclic_tx,
    attenuation=tx_attenuation,
    channel=tx_channel,
)

# ====================== CREATE SIGNAL ======================
iq_tx = 


# ====================== TX ======================
sdr.tx_destroy_buffer()  # Clear TX buffer
sdr.tx(iq_tx)            # Load buffer and transmit -> Keeps going due to cyclic = True


# ===================== DEBUGGING ===================
print_pluto_config(sdr)  # Prints pluto parameters


# ===================== MAIN ========================
def main():
    # Implement code here
    pass

if __name__ == "__main__":
    main()

