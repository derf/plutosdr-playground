#!/usr/bin/env python3

import adi
import numpy as np
import sys
import time
from progress.bar import Bar

from scipy.io import wavfile
import scipy.signal


def am_to_pm(samples, scale=2**14):
    phase_prev = 0
    bar = Bar("AM to FM", max=samples.shape[0] // 1e6)
    bar.start()

    # fm_samples = am_samples / scale * np.pi * 12500 / 1e6

    fm_samples = np.empty(samples.shape, dtype=complex)
    for i, am_sample in enumerate(samples):
        phase_inc = am_sample / scale
        phase_prev += phase_inc * np.pi * 12500 / 1e6
        # phase_prev %= 2 * np.pi
        sample_i = np.cos(phase_prev)
        sample_q = np.sin(phase_prev)
        fm_samples[i] = complex(sample_i * scale, sample_q * scale)
        if (i % 1e6) == 0:
            bar.next()
    bar.finish()

    return fm_samples


if __name__ == "__main__":

    n_samples_per_tx = 10_000
    sdr_sample_rate = 1e6

    wav_sample_rate, data = wavfile.read(sys.argv[1])

    # Stereo → Mono
    if len(data.shape) == 2:
        data = data[:, 0]

    # 16-bit signed to 15-bit signed
    data = data * 0.5

    # Only first few seconds
    data = data[: int(2e6)]

    # Convert to 1 MHz
    print("Resampling …")
    samples = scipy.signal.resample(
        data, int(data.shape[0] * (sdr_sample_rate / wav_sample_rate))
    )

    print("Band Pass …")
    bb = scipy.signal.firwin(41, (10, 20000), pass_zero=False, fs=sdr_sample_rate)
    samples = scipy.signal.lfilter(bb, [1], samples)

    fm_samples = am_to_pm(samples)

    sdr = adi.Pluto("ip:192.168.2.1")
    sample_rate = 1e6
    center_freq = 144600e3

    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(
        sample_rate
    )  # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    sdr.tx_hardwaregain_chan0 = (
        0  # Increase to increase tx power, valid range is -90 to 0 dB
    )

    print("SDR configured, waiting 5 seconds before beginning transmission …")

    time.sleep(5)

    bar = Bar("TX", max=fm_samples.shape[0] // n_samples_per_tx)
    bar.start()

    for i in range(fm_samples.shape[0] // n_samples_per_tx):
        sdr.tx(fm_samples[i * n_samples_per_tx : (i + 1) * n_samples_per_tx])
        bar.next()
    bar.finish()
