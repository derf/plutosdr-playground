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
import multiprocessing
import numpy as np
import scipy.io
import scipy.signal
import sys
import time
from progress.bar import Bar


def tx(queue, connection, sample_rate, carrier, tx_power, n_samples_per_tx, n_blocks):

    sdr = adi.Pluto(connection)

    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(carrier)
    sdr.tx_hardwaregain_chan0 = tx_power

    time.sleep(1)

    bar = Bar("TX", max=n_blocks)
    for i in range(n_blocks):
        samples = queue.get()
        if samples is None:
            break
        sdr.tx(samples)
        bar.next()
    bar.finish()

    # The device seems to continue transmitting -something- even after the final tx call is over.
    # Set its TX power to the minimum to make these emissions harmless.
    sdr.tx_hardwaregain_chan0 = -89


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

    wav_sample_rate, wav_data = scipy.io.wavfile.read(args.input)

    # Stereo → Mono
    if len(wav_data.shape) == 2:
        wav_data = wav_data[:, 0]

    # Normalize 16-bit signed input to -1 … 1
    wav_data = wav_data / 2**15

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    tx_process = context.Process(
        target=tx,
        args=(
            queue,
            args.pluto_connection,
            sdr_sample_rate,
            args.carrier * 1e6,
            args.tx_power,
            n_samples_per_tx,
            wav_data.shape[0] // 96_000,
        ),
    )
    tx_process.start()

    bb = scipy.signal.firwin(
        41, (20, fm_deviation), pass_zero=False, fs=sdr_sample_rate
    )

    phase_offset = 0

    for i in range(wav_data.shape[0] // 96_000):
        data = wav_data[i * 96_000 : (i + 1) * 96_000]

        # Convert to 1 MHz
        samples = scipy.signal.resample(
            data, int(data.shape[0] * (sdr_sample_rate / wav_sample_rate))
        )
        samples = scipy.signal.lfilter(bb, [1], samples)

        fm_samples = samples * np.pi * fm_deviation / sdr_sample_rate
        fm_samples[0] += phase_offset

        phase_integral = np.cumsum(fm_samples)
        phase_offset = phase_integral[-1] % (2 * np.pi)

        fm_samples = np.exp(1j * phase_integral)

        # PlutoSDR expects 15-bit signed input
        fm_samples *= 2**14

        queue.put(fm_samples)

    tx_process.join()
