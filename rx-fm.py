#!/usr/bin/env python3

import adi
import numpy as np
from scipy.io import wavfile
import scipy.signal
import time
from progress.bar import Bar
import sys

if __name__ == "__main__":

    sdr_sample_rate = 1e6  # Hz
    wav_sample_rate = 48000  # Hz
    center_freq = sys.argv[1]  # Hz
    n_samples = 100_000  # number of samples returned per call to rx()

    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 70.0
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sdr_sample_rate)
    sdr.rx_rf_bandwidth = int(sdr_sample_rate)
    sdr.rx_buffer_size = n_samples

    time.sleep(2)

    iq_samples = list()

    bar = Bar("RX", max=100)
    bar.start()

    for i in range(100):
        iq_samples.extend(sdr.rx())  # receive samples off Pluto
        bar.next()

    bar.finish()

    iq_samples = np.array(iq_samples)

    wav_samples = 0.5 * np.angle(iq_samples[:-1] * np.conj(iq_samples[1:]))

    bb = scipy.signal.firwin(41, cutoff=12_000, pass_zero=False, fs=sdr_sample_rate)
    wav_samples = scipy.signal.lfilter(bb, [1], wav_samples)

    wav_samples = scipy.signal.resample(
        wav_samples, int(wav_samples.shape[0] * (wav_sample_rate / sdr_sample_rate))
    )

    out_samples = np.array(wav_samples * 2**16, dtype=np.int16)
    print(np.min(out_samples), np.max(out_samples))
    wavfile.write("/tmp/out.wav", wav_sample_rate, out_samples)
