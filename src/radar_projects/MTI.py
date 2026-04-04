#!/usr/bin/env python3
"""
Pluto SDR indoor MTI motion detector
------------------------------------

Purpose
- Reliable indoor motion detection
- No range measurement
- Real-time movement strength indicator plot

Key choices
- Pluto SDR
- Sample rate = 5 MHz
- Continuous-wave transmit tone
- Adaptive clutter removal for MTI-like behavior
- Headless-safe plotting by default (writes PNG instead of opening Qt)

Requirements
    pip install pyadi-iio numpy matplotlib

Examples
    python MTI.py
    python MTI.py --uri ip:192.168.2.1
    python MTI.py --backend TkAgg
"""

import argparse
import signal
import sys
import time
from collections import deque

import numpy as np

# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_URI = "ip:192.168.2.1"

SAMPLE_RATE = int(5e6)          # requested by user
CENTER_FREQ = int(3.5e9)       # realistic indoor ISM-band choice
TX_TONE_HZ = int(100e3)         # CW offset tone
RX_BUFFER_SIZE = 32768          # practical block size
RF_BANDWIDTH = int(2e6)         # enough for this CW / Doppler application

TX_GAIN_DB = -45.0              # conservative indoor TX level
RX_GAIN_DB = 40.0               # stable starting point indoors

DECIMATION = 100                # 5 MHz -> 50 kHz slow-time stream
LPF_LEN = 64                    # simple LPF before decimation

CLUTTER_ALPHA = 0.005           # slower = stronger static rejection
DISPLAY_HISTORY = 400
INDICATOR_SMOOTH = 0.12

PNG_SAVE_EVERY = 5              # save plot every N frames
PNG_FILENAME = "motion_indicator.png"

# Simple motion decision aid
NOISE_FLOOR_ALPHA = 0.01
MOTION_MARGIN_DB = 6.0          # indicator must exceed learned floor by this much


def parse_args():
    parser = argparse.ArgumentParser(description="Pluto SDR MTI motion detector")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Pluto URI, e.g. ip:192.168.2.1")
    parser.add_argument("--center-freq", type=float, default=CENTER_FREQ, help="RF center frequency in Hz")
    parser.add_argument("--tx-tone", type=float, default=TX_TONE_HZ, help="TX tone offset in Hz")
    parser.add_argument("--tx-gain", type=float, default=TX_GAIN_DB, help="TX hardware gain in dB")
    parser.add_argument("--rx-gain", type=float, default=RX_GAIN_DB, help="RX manual gain in dB")
    parser.add_argument("--backend", default="Agg", help="Matplotlib backend, e.g. Agg or TkAgg")
    parser.add_argument("--png", default=PNG_FILENAME, help="PNG file to update")
    return parser.parse_args()


args = parse_args()

# Must set backend before importing pyplot
import matplotlib
matplotlib.use(args.backend)
import matplotlib.pyplot as plt

import adi


def generate_tx_tone(fs: int, tone_hz: int, n: int, amplitude: float = 0.25) -> np.ndarray:
    """
    Generate a complex TX waveform for cyclic transmission.
    """
    t = np.arange(n, dtype=np.float64) / fs
    iq = amplitude * np.exp(1j * 2.0 * np.pi * tone_hz * t)

    # Scale to a practical level for Pluto complex samples
    iq = iq * (2 ** 14)
    return iq.astype(np.complex64)


def moving_average_complex(x: np.ndarray, n: int) -> np.ndarray:
    """
    Simple boxcar LPF for I and Q separately.
    """
    if n <= 1:
        return x
    kernel = np.ones(n, dtype=np.float64) / n
    xr = np.convolve(np.real(x), kernel, mode="same")
    xi = np.convolve(np.imag(x), kernel, mode="same")
    return (xr + 1j * xi).astype(np.complex64)


def robust_indicator(x: np.ndarray) -> float:
    """
    Compute a stable scalar motion indicator from clutter-cancelled baseband.
    """
    power = np.mean(np.abs(x) ** 2) + 1e-12
    rms = np.sqrt(power)
    return 20.0 * np.log10(rms + 1e-12)


def connect_pluto(uri: str):
    """
    Connect to Pluto using an explicit URI to avoid auto-discovery issues.
    """
    try:
        sdr = adi.Pluto(uri=uri)
    except TypeError:
        # Some environments accept positional URI
        sdr = adi.Pluto(uri)
    return sdr


def setup_pluto(sdr, center_freq: int, tx_gain: float, rx_gain: float):
    """
    Configure Pluto for single-channel CW radar use.
    """
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_lo = int(center_freq)
    sdr.tx_lo = int(center_freq)
    sdr.rx_rf_bandwidth = RF_BANDWIDTH
    sdr.tx_rf_bandwidth = RF_BANDWIDTH
    sdr.rx_buffer_size = RX_BUFFER_SIZE

    # Explicit single-channel setup
    try:
        sdr.rx_enabled_channels = [0]
    except Exception:
        pass
    try:
        sdr.tx_enabled_channels = [0]
    except Exception:
        pass

    # Manual gain is more stable for motion indication than AGC
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(rx_gain)
    sdr.tx_hardwaregain_chan0 = float(tx_gain)

    sdr.tx_cyclic_buffer = True


def make_ddc(fs: int, tone_hz: int, block_len: int) -> np.ndarray:
    n = np.arange(block_len, dtype=np.float64)
    lo = np.exp(-1j * 2.0 * np.pi * tone_hz * n / fs)
    return lo.astype(np.complex64)


def ensure_1d_complex(x):
    """
    Normalize Pluto RX output to a 1D complex numpy array.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.array([arr], dtype=np.complex64)
    if arr.ndim == 1:
        return arr.astype(np.complex64)
    # Some drivers return shape like (channels, samples)
    return np.asarray(arr[0]).astype(np.complex64)


def main():
    stop = {"flag": False}

    def handle_sigint(sig, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        sdr = connect_pluto(args.uri)
    except Exception as e:
        print(f"ERROR: Could not connect to Pluto at URI '{args.uri}'", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        setup_pluto(
            sdr=sdr,
            center_freq=int(args.center_freq),
            tx_gain=float(args.tx_gain),
            rx_gain=float(args.rx_gain),
        )
    except Exception as e:
        print("ERROR: Pluto configuration failed", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        sys.exit(1)

    tx_len = 65536
    tx_iq = generate_tx_tone(
        fs=SAMPLE_RATE,
        tone_hz=int(args.tx_tone),
        n=tx_len,
        amplitude=0.20,
    )

    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    try:
        sdr.tx(tx_iq)
    except Exception as e:
        print("ERROR: Failed to start TX waveform", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        sys.exit(1)

    ddc = make_ddc(SAMPLE_RATE, int(args.tx_tone), RX_BUFFER_SIZE)

    clutter_ref = None
    indicator_smoothed = -120.0
    learned_floor = None
    history = deque(maxlen=DISPLAY_HISTORY)
    thresh_history = deque(maxlen=DISPLAY_HISTORY)
    frame_counter = 0
    last_print_time = 0.0

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 5))
    indicator_line, = ax.plot([], [], lw=2, label="Movement strength")
    threshold_line, = ax.plot([], [], lw=1, linestyle="--", label="Motion threshold")
    ax.set_title("Pluto SDR MTI Motion Strength Indicator")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Level (dB, relative)")
    ax.grid(True)
    ax.legend(loc="upper right")

    text_box = ax.text(
        0.02, 0.97, "",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    print("Pluto SDR MTI motion detector running")
    print(f"URI               : {args.uri}")
    print(f"Sample rate       : {SAMPLE_RATE/1e6:.3f} MS/s")
    print(f"Center frequency  : {args.center_freq/1e9:.6f} GHz")
    print(f"TX tone           : {args.tx_tone/1e3:.1f} kHz")
    print(f"RX gain           : {args.rx_gain:.1f} dB")
    print(f"TX gain           : {args.tx_gain:.1f} dB")
    print(f"Plot backend      : {args.backend}")
    print(f"PNG output        : {args.png}")
    print("Press Ctrl+C to stop")

    try:
        while not stop["flag"]:
            try:
                rx = sdr.rx()
            except Exception as e:
                print(f"RX error: {e}", file=sys.stderr)
                time.sleep(0.05)
                continue

            rx = ensure_1d_complex(rx)

            if len(rx) != RX_BUFFER_SIZE:
                # Rebuild DDC if driver returns a different size
                local_ddc = make_ddc(SAMPLE_RATE, int(args.tx_tone), len(rx))
            else:
                local_ddc = ddc

            # 1) Mix the known TX tone to near DC
            bb = rx * local_ddc

            # 2) LPF and decimate
            bb = moving_average_complex(bb, LPF_LEN)
            bb = bb[::DECIMATION]

            # 3) Remove residual mean
            bb = bb - np.mean(bb)

            # 4) Adaptive clutter cancellation
            if clutter_ref is None:
                clutter_ref = bb.copy()
                mti = np.zeros_like(bb)
            else:
                if len(clutter_ref) != len(bb):
                    clutter_ref = bb.copy()
                clutter_ref = ((1.0 - CLUTTER_ALPHA) * clutter_ref + CLUTTER_ALPHA * bb).astype(np.complex64)
                mti = bb - clutter_ref

            # 5) Movement strength indicator
            indicator = robust_indicator(mti)
            indicator_smoothed = (
                (1.0 - INDICATOR_SMOOTH) * indicator_smoothed
                + INDICATOR_SMOOTH * indicator
            )

            # 6) Learn quiet background floor slowly
            if learned_floor is None:
                learned_floor = indicator_smoothed
            else:
                if indicator_smoothed < learned_floor + 2.0:
                    learned_floor = (
                        (1.0 - NOISE_FLOOR_ALPHA) * learned_floor
                        + NOISE_FLOOR_ALPHA * indicator_smoothed
                    )

            threshold = learned_floor + MOTION_MARGIN_DB
            motion_detected = indicator_smoothed > threshold

            history.append(indicator_smoothed)
            thresh_history.append(threshold)
            frame_counter += 1

            now = time.time()
            if now - last_print_time > 0.5:
                state = "MOTION" if motion_detected else "quiet"
                print(
                    f"indicator={indicator_smoothed:7.2f} dB | "
                    f"floor={learned_floor:7.2f} dB | "
                    f"threshold={threshold:7.2f} dB | "
                    f"{state}"
                )
                last_print_time = now

            # Update plot
            y = np.array(history, dtype=np.float64)
            yt = np.array(thresh_history, dtype=np.float64)
            x = np.arange(len(y), dtype=np.float64)

            indicator_line.set_data(x, y)
            threshold_line.set_data(x, yt)

            ax.set_xlim(0, max(DISPLAY_HISTORY, len(y)))
            if len(y) > 0:
                ymin = min(np.min(y), np.min(yt)) - 3
                ymax = max(np.max(y), np.max(yt)) + 3
            else:
                ymin, ymax = -90, -20
            if ymax - ymin < 10:
                mid = 0.5 * (ymax + ymin)
                ymin = mid - 5
                ymax = mid + 5
            ax.set_ylim(ymin, ymax)

            status = "MOTION DETECTED" if motion_detected else "No motion"
            text_box.set_text(
                f"{status}\n"
                f"Indicator: {indicator_smoothed:6.1f} dB\n"
                f"Threshold: {threshold:6.1f} dB\n"
                f"Center: {args.center_freq/1e9:.3f} GHz\n"
                f"TX tone: {args.tx_tone/1e3:.0f} kHz"
            )

            # Save PNG periodically for headless environments
            if frame_counter % PNG_SAVE_EVERY == 0:
                fig.savefig(args.png, dpi=120, bbox_inches="tight")

            # If using an interactive backend, refresh the window
            if args.backend.lower() != "agg":
                plt.pause(0.001)

    finally:
        try:
            fig.savefig(args.png, dpi=120, bbox_inches="tight")
        except Exception:
            pass

        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

        print(f"Stopped. Last plot saved to: {args.png}")


if __name__ == "__main__":
    main()