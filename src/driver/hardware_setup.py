import adi
import numpy as np

DEFAULT_URI = "ip:192.168.2.1"
DEFAULT_SAMPLE_RATE = 2_500_000
DEFAULT_RX_BUFFER_SIZE = 32768
DEFAULT_RX_BANDWIDTH = DEFAULT_SAMPLE_RATE
DEFAULT_RX_LO = 2_400_000_000
DEFAULT_RX_GAIN_MODE = "slow_attack"
DEFAULT_RX_CHANNEL = 0


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
    else:
        sdr.gain_control_mode_chan1 = gain_mode

    return sdr


def receive_samples(sdr: adi.Pluto) -> np.ndarray:
    """Return complex IQ samples from the currently configured RX path."""
    return np.asarray(sdr.rx())