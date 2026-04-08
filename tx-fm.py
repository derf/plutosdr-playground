#!/usr/bin/env python3

import adi
import argparse
import numpy as np
import scipy.io
import scipy.signal
import sys
import time
from progress.bar import Bar


def am_to_pm(samples, fm_bandwidth, sdr_sample_rate, scale=2**14):
    fm_samples = samples / scale * np.pi * fm_bandwidth / sdr_sample_rate
    phase_prev = np.cumsum(fm_samples)
    fm_samples = (np.cos(phase_prev) + 1j * np.sin(phase_prev)) * scale

    return fm_samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--carrier", type=float, default=144.6, help="FM carrier frequency [MHz]"
    )
    parser.add_argument(
        "--fm-bandwidth", type=float, default=12.5, help="FM bandwidth [kHz]"
    )
    parser.add_argument(
        "--samples-per-tx",
        type=int,
        default=100_000,
        help="# of samples per adi.Pluto TX call",
    )
    parser.add_argument(
        "--pluto-connection",
        type=str,
        default="ip:192.168.2.1",
        help="Connection to PlutoSDR",
    )
    parser.add_argument("--tx-power", type=int, help="TX Power [dB]; 0 ≙ 10 mW")
    parser.add_argument(
        "input", metavar="input.wav", type=str, help="Input file for transmission"
    )

    args = parser.parse_args()

    fm_bandwidth = int(args.fm_bandwidth * 1e3)
    n_samples_per_tx = args.samples_per_tx
    sdr_sample_rate = 1e6

    wav_sample_rate, data = scipy.io.wavfile.read(args.input)

    # Stereo → Mono
    if len(data.shape) == 2:
        data = data[:, 0]

    # 16-bit signed to 15-bit signed
    data = data * 0.5

    # Only first few seconds
    data = data[: int(1e6)]

    # Convert to 1 MHz
    print("Resampling …")
    samples = scipy.signal.resample(
        data, int(data.shape[0] * (sdr_sample_rate / wav_sample_rate))
    )

    print("Band Pass …")
    bb = scipy.signal.firwin(
        41, (20, fm_bandwidth / 2 - 1), pass_zero=False, fs=fm_bandwidth
    )
    samples = scipy.signal.lfilter(bb, [1], samples)

    fm_samples = am_to_pm(samples, fm_bandwidth, sdr_sample_rate)
    del samples

    sdr = adi.Pluto(args.pluto_connection)

    sdr.sample_rate = int(sdr_sample_rate)
    sdr.tx_rf_bandwidth = int(
        sdr_sample_rate
    )  # filter cutoff, just set it to the same as sample rate

    sdr.tx_lo = int(args.carrier * 1e6)

    # Increase to increase tx power, valid range is -90 to 0 dB
    sdr.tx_hardwaregain_chan0 = args.tx_power

    print("SDR configured, waiting 2 seconds before beginning transmission …")

    time.sleep(2)

    bar = Bar("TX", max=fm_samples.shape[0] // n_samples_per_tx)
    bar.start()

    for i in range(fm_samples.shape[0] // n_samples_per_tx):
        sdr.tx(fm_samples[i * n_samples_per_tx : (i + 1) * n_samples_per_tx])
        bar.next()
    bar.finish()
