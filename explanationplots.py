from scipy import signal, stats
import numpy as np


def explain_freq_plot(sig, fs=300):
    fourierTransform = np.fft.fft(sig) / len(sig)
    fourierTransform = fourierTransform[range(int(len(sig) / 2))]
    fourierTransform = abs(fourierTransform)

    tpCount = len(sig)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / fs
    frequencies = values / timePeriod

    f, Pxx_den = signal.periodogram(sig, 300)
    max_y = max(Pxx_den)  # Find the maximum y value
    max_amp = max(abs(fourierTransform))
    max_x = frequencies[fourierTransform.argmax()]

    return max_x, fourierTransform[0:1500], frequencies[0:1500]  # Display only till 50 Hz


def distribution_score_for_explain(r_peaks, sig):
    first_peak = r_peaks[0]
    last_peak = r_peaks[-1]
    sig = sig[first_peak: last_peak + 1]
    samples_per_beat = len(sig) / (len(r_peaks) - 1)
    # Set threshold, Hardcoded for now, Need to automate later
    lower_thresh = samples_per_beat - 30
    higher_thresh = samples_per_beat + 30
    cumdiff = np.diff(r_peaks)
    # Number of R-R durations lie inside threshold
    # error1 = ([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
    slow1 = np.where(cumdiff < lower_thresh)
    fast1 = np.where(cumdiff > higher_thresh)
    slow1 = [r_peaks[x + 1] for x in slow1]  # returns index of abnormality
    fast1 = [r_peaks[x + 1] for x in fast1]  # returns index of abnormality

    max1 = np.quantile(cumdiff, 0.5)
    # max1 = x[np.argmax(y)]
    # print(max1)
    lower_thresh = max1 - 30
    higher_thresh = max1 + 30
    # Number of R-R durations lie inside threshold
    len_center_idx = len([i for i in cumdiff if i < lower_thresh and i > higher_thresh])
    score2 = len_center_idx / len(cumdiff)
    slow2 = np.where(cumdiff < lower_thresh)
    fast2 = np.where(cumdiff > higher_thresh)
    slow2 = [r_peaks[x + 1] for x in slow2]  # returns index of abnormality
    fast2 = [r_peaks[x + 1] for x in fast2]  # returns index of abnormality

    return slow1, fast1, slow2, fast2