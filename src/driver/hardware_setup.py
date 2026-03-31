import adi
import numpy as np

DEFAULT_URI = "ip:192.168.2.1"
DEFAULT_SAMPLE_RATE = 2_500_000
DEFAULT_RX_BUFFER_SIZE = 32768
DEFUALT_TX_BUFFER_SIZE = 32768
DEFAULT_RX_BANDWIDTH = DEFAULT_SAMPLE_RATE
DEFAULT_TX_BANDWIDTH = DEFAULT_SAMPLE_RATE
DEFAULT_RX_LO = 2_400_000_000
DEFAULT_TX_LO = 2_400_000_000
DEFAULT_RX_GAIN_MODE = "slow_attack"
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
    return sdr


def configure_rx(
    sdr: adi.Pluto,
    center_freq: int = DEFAULT_RX_LO,
    bandwidth: int = DEFAULT_RX_BANDWIDTH,
    buffer_size: int = DEFAULT_RX_BUFFER_SIZE,
    gain_mode: str = DEFAULT_RX_GAIN_MODE,
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
    elif channel == 1:
        sdr.gain_control_mode_chan1 = gain_mode

    return sdr

def configure_tx(
        sdr: adi.Pluto,
        center_freq: int = DEFAULT_RX_LO,
        bandwidth: int = DEFAULT_TX_BANDWIDTH,
        buffer_size: int = DEFAULT_TX_BANDWIDTH,
        cyclic: bool = DEFAULT_CYCLIC_TX,
        attenuation: int = DEFAULT_TX_ATTENUATION,
        channel: int = DEFAULT_TX_CHANNEL,
) -> adi.Pluto:
    channel = int(channel)
    if channel not in (0, 1):
        raise ValueError(f"Unsupported TX channel: {channel}")
    
    sdr.tx_lo = center_freq
    sdr.tx_rf_bandwidth = bandwidth
    sdr.tx_cyclic_buffer = cyclic
    sdr.tx_enabled_channels = [channel]
    sdr.tx_buffer_size = int(DEFUALT_TX_BUFFER_SIZE)
    
    if channel == 0:
        sdr.gain_control_mode_chan0 = attenuation
    elif channel == 1:
        sdr.gain_control_mode_chan1 = attenuation


def receive_samples(sdr: adi.Pluto) -> np.ndarray:
    """Return complex IQ samples from the currently configured RX path."""
    return np.asarray(sdr.rx())