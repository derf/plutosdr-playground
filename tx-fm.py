#!/usr/bin/env python3

import adi
import numpy as np
import sys
import time
from progress.bar import Bar

from scipy.io import wavfile
import scipy.signal


def am_to_pm(samples, fm_bandwidth, sdr_sample_rate, scale=2**14):
    fm_samples = samples / scale * np.pi * fm_bandwidth / sdr_sample_rate
    phase_prev = np.cumsum(fm_samples)
    fm_samples = (np.cos(phase_prev) + 1j * np.sin(phase_prev)) * scale

    return fm_samples


if __name__ == "__main__":

    fm_bandwidth = 12_500
    n_samples_per_tx = 100_000
    sdr_sample_rate = 1e6

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <freq (MHz)> <wav file>")
        sys.exit(1)

    wav_sample_rate, data = wavfile.read(sys.argv[2])

    # Stereo → Mono
    if len(data.shape) == 2:
        data = data[:, 0]

    # 16-bit signed to 15-bit signed
    data = data * 0.5

    # Only first few seconds
    data = data[: int(10e6)]

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

    sdr = adi.Pluto("ip:192.168.2.1")
    sample_rate = 1e6
    center_freq = float(sys.argv[1]) / 1e6

    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(
        sample_rate
    )  # filter cutoff, just set it to the same as sample rate

    sdr.tx_lo = int(center_freq)

    # Increase to increase tx power, valid range is -90 to 0 dB
    sdr.tx_hardwaregain_chan0 = -10

    print("SDR configured, waiting 2 seconds before beginning transmission …")

    time.sleep(2)

    bar = Bar("TX", max=fm_samples.shape[0] // n_samples_per_tx)
    bar.start()

    for i in range(fm_samples.shape[0] // n_samples_per_tx):
        sdr.tx(fm_samples[i * n_samples_per_tx : (i + 1) * n_samples_per_tx])
        bar.next()
    bar.finish()
