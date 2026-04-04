#!/usr/bin/env python3

import adi
import numpy as np
import time
from progress.bar import Bar

from scipy.io import wavfile

if __name__ == "__main__":

    n_samples_per_tx = 10_000
    sdr_sample_rate = 1e6

    wav_sample_rate, data = wavfile.read("/tmp/test.wav")

    # Stereo → Mono
    if len(data.shape) == 2:
        data = data[:, 0]

    # 16-bit signed to 15-bit signed
    data = data * 0.5

    samples = np.repeat(data, sdr_sample_rate // wav_sample_rate)

    # samples = np.empty((data.shape[0] * 23,))
    # bar = Bar("Converting", max=samples.shape[0] // 10000 )
    # bar.start()
    # for i in range(samples.shape[0]):
    #    samples[i] = data[i//23]
    #    if (i % 10000) == 0:
    #        bar.next()
    #
    # bar.finish()

    sdr = adi.Pluto("ip:192.168.2.1")
    sample_rate = 1e6
    center_freq = 144600e3

    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(
        sample_rate
    )  # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    sdr.tx_hardwaregain_chan0 = (
        -20
    )  # Increase to increase tx power, valid range is -90 to 0 dB

    print("SDR configured, waiting 5 seconds before beginning transmission …")

    time.sleep(5)

    bar = Bar("TX", max=samples.shape[0] // n_samples_per_tx)
    bar.start()

    for i in range(samples.shape[0] // n_samples_per_tx):
        sdr.tx(samples[i * n_samples_per_tx : (i + 1) * n_samples_per_tx])
        bar.next()
    bar.finish()
