import time
import numpy as np
import iio
import matplotlib

PLOT_ENABLE = True
PLOT_EVERY = 200
PRINT_EVERY = 100
CHECK_STATUS_EVERY = 10

if PLOT_ENABLE:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

from driver.hardware_setup import *
from driver.waveforms import *
from driver.plots import *
from driver.signal_processing import *

# ======== PLUTO PARAMETERS =========
uri = "ip:192.168.2.1"
sample_rate = 5_000_000
ts = 1 / sample_rate
rx_buffer_size = 2**18

rx_bandwidth = sample_rate
tx_bandwidth = sample_rate
rx_lo = 2.4e9
tx_lo = 2.4e9
rx_gain_mode = "manual"
rx_gain = 10
rx_channel = 0
tx_channel = 0
cyclic_tx = True
tx_attenuation = -20

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
iq_tx = make_chirp(fs=sample_rate, f0=-2.5e6, f1=2.5e6, n=2048)

sdr.tx_destroy_buffer()
sdr.tx(iq_tx)

print_pluto_config(sdr)

# ======== LOW-LEVEL IIO STATUS MONITORING ========
ctx = iio.Context(uri)
rx_dma = ctx.find_device("cf-ad9361-lpc")

if rx_dma is None:
    raise RuntimeError("Could not find RX DMA device: cf-ad9361-lpc")

UI_STATUS = 0x80000088
RX_OVF_BIT = 0x04

def clear_rx_overflow():
    rx_dma.reg_write(UI_STATUS, RX_OVF_BIT)

def check_rx_overflow():
    stat = rx_dma.reg_read(UI_STATUS)
    hit = bool(stat & RX_OVF_BIT)
    if hit:
        clear_rx_overflow()
    return hit

clear_rx_overflow()

def setup_plot():
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=1, color="lime")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("dBFS")
    ax.set_title("Live FFT Spectrum")
    ax.set_ylim(-120, 0)
    ax.grid(True)
    ax.set_facecolor("black")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))
    plt.ion()
    plt.show(block=False)
    return fig, ax, line

def update_plot(fig, ax, line, rx_samples):
    xf, s_dbfs = compute_fft_dbfs(rx_samples, ts)
    xf_rf = xf + rx_lo / 1e6
    line.set_data(xf_rf, s_dbfs)
    ax.set_xlim(xf_rf[0], xf_rf[-1])
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def main():
    print("Running... press Ctrl+C to stop")

    fig = ax = line = None
    if PLOT_ENABLE:
        fig, ax, line = setup_plot()

    rx_overflow_count = 0
    loop_count = 0
    sample_count = 0
    t0 = time.time()

    try:
        while True:
            rx_samples = sdr.rx()
            n = len(rx_samples)
            sample_count += n
            loop_count += 1

            if loop_count % CHECK_STATUS_EVERY == 0:
                if check_rx_overflow():
                    rx_overflow_count += 1

            if PLOT_ENABLE and (loop_count % PLOT_EVERY == 0):
                update_plot(fig, ax, line, rx_samples)

            if loop_count % PRINT_EVERY == 0:
                elapsed = time.time() - t0
                ms_per_loop = 1000 * elapsed / loop_count
                ms_per_buffer = 1000 * n / sample_rate
                effective_msps = (sample_count / elapsed) / 1e6
                margin = ms_per_buffer - ms_per_loop

                print(
                    f"[INFO] loops={loop_count}, "
                    f"elapsed={elapsed:.2f}s, "
                    f"rx_ovf={rx_overflow_count}, "
                    f"loop_time={ms_per_loop:.2f} ms, "
                    f"buffer_time={ms_per_buffer:.2f} ms, "
                    f"margin={margin:.2f} ms, "
                    f"effective_rx={effective_msps:.2f} MS/s"
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        sdr.tx_destroy_buffer()
        if PLOT_ENABLE and fig is not None:
            plt.ioff()
            plt.close(fig)

if __name__ == "__main__":
    main()