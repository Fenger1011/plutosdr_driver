import os
import matplotlib
matplotlib.use("QtAgg")
from datetime import datetime

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Import driver
from plutosdr_driver import *


# ====================== PLUTO PARAMETERS ======================
uri = "ip:192.168.2.1"

fs = 521e3                          # Sample rate
ts = 1 / fs                         # Sample period

rx_lo = 2.175e9                     # RX LO
tx_lo = 2.175e9                     # TX LO

lam = 3e8 / tx_lo                   # Wavelength corresponding to TX LO

rx_gain_mode = "manual"             # Gain mode
rx_gain = 30                        # Gain in dB
tx_attenuation = 0                  # Attenuation in dB

rx_channel = 0                      # Active RX channel
tx_channel = 0                      # Active TX channel
cyclic_tx = True                    # TX cyclic buffer True

rx_bandwidth = fs                   # RX bandwidth
tx_bandwidth = fs                   # TX bandwidth

fs_decim_target = 16_240            # Sample rate after decimation
ts_decim = 1 / fs_decim_target      # Sample period after decimation

N_fft = 4096                        # Number of bins in FFT (corresponding to dwell time)

overlap_frac = 0.25                     # Each FFT frame overlaps previous by 25%
overlap_len = int(N_fft * overlap_frac) # Length of FFT frame which overlaps
hop_len = N_fft - overlap_len           # Length of FFT which is NOT being overlapped

M = int(round(fs / fs_decim_target))    # Decimation factor (521e3/16240 = 32)

# Acquire only enough raw samples each loop to produce one new hop
rx_buffer_size = int(np.ceil(hop_len * fs / fs_decim_target))


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
iq_tx = (2**14) * np.ones(4096, dtype=np.complex64)  # Constant complex baseband signal


# ====================== TX ======================
sdr.tx_destroy_buffer()  # Clear TX buffer
sdr.tx(iq_tx)            # Load buffer and transmit -> Keeps going due to cyclic = True


# ===================== DEBUGGING ===================
print_pluto_config(sdr)
print(f"RX buffer size (raw): {rx_buffer_size} samples")
print(f"Decimation rate:      {M}")
print(f"FFT frame length:     {N_fft} samples")
print(f"Overlap:              {overlap_len} samples ({overlap_frac*100:.0f}%)")
print(f"Hop length:           {hop_len} samples")


# ====================== MAIN LOOP ======================
def main():
    fs_decim_actual = fs / M                # Actual decimated sample rate
    ts_decim_actual = 1 / fs_decim_actual

    # Doppler search limits
    f_min = -812
    f_max = 812
    guard = 10

    # Detection settings
    min_snr_db = 10.0
    min_prominence_db = 8.0
    bg_alpha = 0.98
    warmup_frames = 15

    # Logging setup -> Saves in /plots
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_dir = os.path.join(base_dir, "plots")

    # Ensure folder exists
    os.makedirs(log_dir, exist_ok=True)

    # Full log file path
    log_filename = os.path.join(log_dir, "doppler_log.csv")

    print(f"Logging to: {log_filename}")

    # Check if file exists BEFORE opening
    file_exists = os.path.exists(log_filename)

    # Open file
    log_file = open(log_filename, "a", buffering=1)

    # Write header if new or empty
    if not file_exists or os.path.getsize(log_filename) == 0:
        log_file.write("timestamp,f_peak_hz,velocity_m_s,velocity_kmh,snr_db\n")

    # Streaming high-pass filter in decimated domain
    sos_hp = signal.butter(4, 20, btype="highpass", fs=fs_decim_actual, output="sos")

    # Filter state for complex streaming -> Filters I and Q separately
    zi_hp_i = signal.sosfilt_zi(sos_hp) * 0.0
    zi_hp_q = signal.sosfilt_zi(sos_hp) * 0.0

    # Rolling FFT history
    fft_history = np.zeros(overlap_len, dtype=np.complex64)
    first_frame = True

    # Background spectrum
    bg_db = None
    frame_counter = 0

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    xf_init = np.linspace(-fs_decim_actual / 2, fs_decim_actual / 2, N_fft, endpoint=False)
    y_init = np.full_like(xf_init, -150.0)

    markerline, stemlines, baseline = ax.stem(xf_init, y_init, basefmt=" ")
    plt.setp(markerline, markersize=3)
    plt.setp(stemlines, linewidth=0.8)

    peak_marker, = ax.plot([], [], "ro", markersize=8, label="Peak")
    ax.axvline(-guard, linestyle="--", linewidth=1, label="Guard band")
    ax.axvline(guard, linestyle="--", linewidth=1)

    ax.set_title("Live FFT")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(-140, 20)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        while True:
            # Receive IQ samples
            rx = sdr.rx()

            # Decimate in streaming-safe way
            rx_decim = signal.decimate(rx, M, ftype="fir", zero_phase=False)

            # Apply HPF on I and Q separately
            rx_i, zi_hp_i = signal.sosfilt(sos_hp, np.real(rx_decim), zi=zi_hp_i)
            rx_q, zi_hp_q = signal.sosfilt(sos_hp, np.imag(rx_decim), zi=zi_hp_q)
            rx_decim = (rx_i + 1j * rx_q).astype(np.complex64)

            # Make sure block length is correct
            if len(rx_decim) < hop_len:
                print(f"Skipping short block: got {len(rx_decim)} decimated samples, need {hop_len}")
                continue
            elif len(rx_decim) > hop_len:
                rx_decim = rx_decim[:hop_len]

            # Build rolling FFT frame
            if first_frame:
                frame = np.zeros(N_fft, dtype=np.complex64)
                frame[-len(rx_decim):] = rx_decim
                first_frame = False
            else:
                frame = np.concatenate((fft_history, rx_decim))

            if len(frame) != N_fft:
                print(f"Frame length error: got {len(frame)}, expected {N_fft}")
                continue

            fft_history = frame[-overlap_len:].copy()

            # Compute FFT
            xf, s_dbfs = compute_fft_dbfs(rx_samples=frame, ts=ts_decim_actual)

            # Background estimate
            if bg_db is None:
                bg_db = s_dbfs.copy()
            else:
                bg_db = bg_alpha * bg_db + (1.0 - bg_alpha) * s_dbfs

            s_det = s_dbfs - bg_db
            frame_counter += 1

            # Search area only outside guard band and within desired range
            idx = np.where(
                ((xf >= f_min) & (xf <= -guard)) |
                ((xf >= guard) & (xf <= f_max))
            )[0]

            if len(idx) == 0:
                print("No valid FFT search region.")
                continue

            search_det = s_det[idx]
            search_raw = s_dbfs[idx]

            k_local = np.argmax(search_det)
            k_peak = idx[k_local]

            f_peak = xf[k_peak]
            peak_raw_db = s_dbfs[k_peak]
            peak_det_db = s_det[k_peak]

            # Estimate noise floor using median
            noise_floor_det_db = np.median(search_det)

            # Use median absolute deviation (MAD) as measure of spread
            mad_det_db = np.median(np.abs(search_det - noise_floor_det_db)) + 1e-12

            # Peak level relative to estimated noise floor
            snr_db = peak_det_db - noise_floor_det_db

            # Peak must stand clearly above the local background/noise
            prominence_ok = peak_det_db > max(min_prominence_db, noise_floor_det_db + 6.0 * mad_det_db)

            # Peak must also pass the minimum SNR threshold
            snr_ok = snr_db >= min_snr_db

            # Ignore detections until the background estimate has stabilized
            warmup_ok = frame_counter >= warmup_frames

            # Motion is declared only if all checks pass
            motion_detected = warmup_ok and prominence_ok and snr_ok

            # ---------------- PLOT UPDATE ----------------
            plot_mask = (xf >= f_min) & (xf <= f_max)
            xf_plot = xf[plot_mask]
            s_plot = s_dbfs[plot_mask]

            markerline.set_data(xf_plot, s_plot)

            segments = [np.array([[x, 0], [x, y]]) for x, y in zip(xf_plot, s_plot)]
            stemlines.set_segments(segments)

            if motion_detected:
                peak_marker.set_data([f_peak], [peak_raw_db])
            else:
                peak_marker.set_data([], [])

            y_min = min(np.min(s_plot) - 5, -140)
            y_max = max(np.max(s_plot) + 5, 20)
            ax.set_ylim(y_min, y_max)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            # ---------------- MOTION DECISION ----------------
            if not warmup_ok:
                print(f"Background warmup... ({frame_counter}/{warmup_frames})")
                continue

            if not motion_detected:
                print(
                    f"No reliable motion | "
                    f"peak_det={peak_det_db:.1f} dB  "
                    f"SNR={snr_db:.1f} dB"
                )
                continue

            v_peak = (lam / 2.0) * f_peak
            timestamp = datetime.now().isoformat(timespec="milliseconds")

            # Log detection without reopening the file each loop
            log_file.write(
                f"{timestamp},"
                f"{f_peak:.2f},"
                f"{v_peak:.4f},"
                f"{v_peak * 3.6:.2f},"
                f"{snr_db:.2f}\n"
            )
            log_file.flush()

            print(
                f"{timestamp} | "
                f"f={f_peak:.1f} Hz  "
                f"v={v_peak:.3f} m/s ({v_peak * 3.6:.1f} km/h)  "
                f"SNR={snr_db:.1f} dB"
            )

    except KeyboardInterrupt:
        print("\nStopping...")

    except Exception as e:
        print("Error:", e)
        raise

    finally:
        try:
            log_file.close()
        except Exception:
            pass

        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()