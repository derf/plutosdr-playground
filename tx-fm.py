#!/usr/bin/env python3

# Copyright (C) 2026 Birte Kristina Friesel
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# See <https://finalrewind.org/interblag/entry/adi-plutosdr-fm-transmission/>

import adi
import argparse
import numpy as np
import scipy.io
import scipy.signal
import sys
import time
from progress.bar import Bar

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--carrier", type=float, default=144.6, help="FM carrier frequency [MHz]"
    )
    parser.add_argument(
        "--fm-deviation",
        type=float,
        default=12.5,
        help="FM deviation [kHz]. Should be determined empirically.",
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
    parser.add_argument(
        "--pluto-sample-rate",
        type=int,
        default=1_000_000,
        help="PlutoSDR sample rate for transmission",
    )
    parser.add_argument(
        "--tx-power",
        type=int,
        choices=list(range(-89, 1)),
        default=-10,
        help="TX Power [dB]; 0 ≙ 10 mW",
    )
    parser.add_argument(
        "input", metavar="input.wav", type=str, help="Input file for transmission"
    )

    args = parser.parse_args()

    fm_deviation = int(args.fm_deviation * 1e3)
    n_samples_per_tx = args.samples_per_tx
    sdr_sample_rate = args.pluto_sample_rate

    wav_sample_rate, data = scipy.io.wavfile.read(args.input)

    # Stereo → Mono
    if len(data.shape) == 2:
        data = data[:, 0]

    # Normalize 16-bit signed input to -1 … 1
    data = data / 2**15

    # Only first few seconds
    data = data[: int(5e6)]

    # Convert to 1 MHz
    print("Resampling …")
    samples = scipy.signal.resample(
        data, int(data.shape[0] * (sdr_sample_rate / wav_sample_rate))
    )

    print("Band Pass …")
    bb = scipy.signal.firwin(
        41, (20, fm_deviation), pass_zero=False, fs=sdr_sample_rate
    )
    samples = scipy.signal.lfilter(bb, [1], samples)

    fm_samples = samples * np.pi * fm_deviation / sdr_sample_rate
    phase_integral = np.cumsum(fm_samples)
    fm_samples = np.exp(1j * phase_integral)

    del samples

    # PlutoSDR expects 15-bit signed input
    fm_samples *= 2**14

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

    # The device seems to continue transmitting -something- even after the final tx call is over.
    # Set its TX power to the minimum to make these emissions harmless.
    sdr.tx_hardwaregain_chan0 = -89
