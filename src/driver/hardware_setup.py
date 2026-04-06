import adi
import numpy as np

DEFAULT_URI = "ip:192.168.2.1"
DEFAULT_SAMPLE_RATE = 2_500_000
DEFAULT_RX_BUFFER_SIZE = 32768
DEFAULT_TX_BUFFER_SIZE = 32768
DEFAULT_RX_BANDWIDTH = DEFAULT_SAMPLE_RATE
DEFAULT_TX_BANDWIDTH = DEFAULT_SAMPLE_RATE
DEFAULT_RX_LO = 2_400_000_000
DEFAULT_TX_LO = 2_400_000_000
DEFAULT_RX_GAIN_MODE = "slow_attack"
DEFAULT_RX_GAIN = 10
DEFAULT_RX_CHANNEL = 0
DEFAULT_TX_CHANNEL = 0
DEFAULT_CYCLIC_TX = False
DEFAULT_TX_ATTENUATION = -10


def create_pluto(uri: str = DEFAULT_URI, sample_rate: int = DEFAULT_SAMPLE_RATE) -> adi.Pluto:
    try:
        sdr = adi.Pluto(uri=uri)
    except Exception as e:
        raise RuntimeError(f"Could not connect to PlutoSDR at {uri}: {e}") from e

    sdr.sample_rate = int(sample_rate)

    # Reduce RX latency
    try:
        sdr._rxadc.set_kernel_buffers_count(1)
    except Exception as e:
        print(f"Warning: could not set RX kernel buffers to 1: {e}")

    return sdr


def configure_rx(
    sdr: adi.Pluto,
    center_freq: int = DEFAULT_RX_LO,
    bandwidth: int = DEFAULT_RX_BANDWIDTH,
    buffer_size: int = DEFAULT_RX_BUFFER_SIZE,
    gain_mode: str = DEFAULT_RX_GAIN_MODE,
    gain: int = DEFAULT_RX_GAIN,
    channel: int = DEFAULT_RX_CHANNEL,
) -> adi.Pluto:
    channel = int(channel)
    if channel not in (0, 1):
        raise ValueError(f"Unsupported RX channel: {channel}")

    sdr.rx_lo = int(center_freq)
    sdr.rx_rf_bandwidth = int(bandwidth)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.rx_enabled_channels = [channel]

    if channel == 0:
        sdr.gain_control_mode_chan0 = gain_mode
        sdr.rx_hardwaregain_chan0 = gain
    elif channel == 1:
        sdr.gain_control_mode_chan1 = gain_mode
        sdr.rx_hardwaregain_chan1 = gain

    return sdr

def configure_tx(
        sdr: adi.Pluto,
        center_freq: int = DEFAULT_TX_LO,
        bandwidth: int = DEFAULT_TX_BANDWIDTH,
        buffer_size: int = DEFAULT_TX_BUFFER_SIZE,
        cyclic: bool = DEFAULT_CYCLIC_TX,
        attenuation: int = DEFAULT_TX_ATTENUATION,
        channel: int = DEFAULT_TX_CHANNEL,
) -> adi.Pluto:
    channel = int(channel)
    if channel not in (0, 1):
        raise ValueError(f"Unsupported TX channel: {channel}")
    
    sdr.tx_lo = int(center_freq)
    sdr.tx_rf_bandwidth = int(bandwidth)
    sdr.tx_cyclic_buffer = cyclic
    sdr.tx_enabled_channels = [channel]
    sdr.tx_buffer_size = int(buffer_size)
    
    if channel == 0:
        sdr.tx_hardwaregain_chan0 = attenuation
    elif channel == 1:
        sdr.tx_hardwaregain_chan1 = attenuation

    return sdr


def receive_samples(sdr: adi.Pluto, buffer_size = DEFAULT_RX_BUFFER_SIZE) -> np.ndarray:
    """Return complex IQ samples from the RX path."""
    sdr.rx_buffer_size = buffer_size    # Makes it possible to change the buffer size

    return np.asarray(sdr.rx())

def print_pluto_config(sdr):
    print("\n=== SDR CONFIG ===")

    print(
        f"TX: ch={sdr.tx_enabled_channels}, LO={sdr.tx_lo}, "
        f"BW={sdr.tx_rf_bandwidth}, cyclic={sdr.tx_cyclic_buffer}, "
        f"buf={sdr.tx_buffer_size}, gain={sdr.tx_hardwaregain_chan0}"
    )

    print(
        f"RX: ch={sdr.rx_enabled_channels}, LO={sdr.rx_lo}, "
        f"BW={sdr.rx_rf_bandwidth}, buf={sdr.rx_buffer_size}, "
        f"gain={sdr.rx_hardwaregain_chan0}, mode={sdr.gain_control_mode_chan0}"
    )

