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
sample_rate = 5e6
ts = 1 / sample_rate
rx_buffer_size = 2**15 #32768
rx_bandwidth = sample_rate
tx_bandwidth = sample_rate
rx_lo = 2.4e9
tx_lo = 2.4e9
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

# ============ CREATE SIGNAL & TX ============
iq_tx = make_tone(fs=sample_rate, tone_hz=1e6, amplitude=2**1)
#iq_tx = make_chirp(fs=sample_rate, f0=-2e6, f1=2e6, n=2**10)

sdr.tx_destroy_buffer()
sdr.tx(iq_tx)

print_pluto_config(sdr)

# ============ MAIN LOOP =============
def main():
    print("Transmitting... press Ctrl+C to stop")

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=1, color="lime")

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("dBFS")
    ax.set_title("Live FFT Spectrum")

    ax.set_ylim(-120, 0) # 0dBFS to -120dBFS

    ax.grid(True)
    ax.set_facecolor("black")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))

    plt.show(block=False)

    try:
        while True:
            rx_samples = sdr.rx()

            xf, s_dbfs = compute_fft_dbfs(rx_samples, ts)

            # Convert to RF frequency
            xf_rf = xf + rx_lo / 1e6

            line.set_data(xf_rf, s_dbfs)

            ax.set_xlim(xf_rf[0], xf_rf[-1])

            fig.canvas.draw_idle()
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")

    except Exception as e:
        print("Error:", e)
        raise

    finally:
        sdr.tx_destroy_buffer()
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()