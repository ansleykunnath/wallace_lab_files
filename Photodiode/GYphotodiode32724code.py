import mne # import the MNE python package for EEG analysis 
import numpy as np # numpy for numerical computing 
from mne.preprocessing import (ICA) # independent component analysis 
from autoreject import AutoReject # for automatic artifact rejection 
import matplotlib # data visualization 
matplotlib.use("TkAgg") # backend for matplotlib to TkAgg
import matplotlib.pyplot as plt

eeg_path = "C://Users//neuro//Documents//VEP//VEP Data/" 
file_name = "PhotoDiode_2024-03-22_10-24-46"
file_vhdr = eeg_path + file_name + ".vhdr"

raw = mne.io.read_raw_brainvision(file_vhdr)

baseline_tmin, baseline_tmax = -0.05, 0
baseline = (baseline_tmin, baseline_tmax)

tmin, tmax = -0.25, 0.25
time_inds = np.where((raw.times >= tmin) & (raw.times <= tmax))[0]

peak_amp = raw.get_data()[0, time_inds].max()
peak_time = raw.times[time_inds][raw.get_data()[0, time_inds].argmax()]

print("Peak amplitude: ", peak_amp)
print("Peak time: ", peak_time)

# Find the time indices corresponding to the time window of interest
tmin, tmax = 0.0, 0.5
time_inds = np.where((raw.times >= tmin) & (raw.times <= tmax))[0]

# Find the peaks in the data
peaks, peak_times = mne.preprocessing.peak_finder(raw.get_data()[0], extrema=1)

# Find the average latencies across all peaks in the data
peak_latencies = peak_times - raw.times[time_inds][0]
avg_latency = np.mean(peak_latencies)

# Print the average latency
print("Average latency: ", avg_latency)

# Plot a histogram of the peak latencies
plt.hist(peak_latencies, bins=20)
plt.xlabel('Latency (s)')
plt.ylabel('Frequency')
plt.show()

# Plot a histogram of the amplitudes found against the latency times
plt.plot(peak_latencies, peaks, 'o')
plt.xlabel('Latency (s)')
plt.ylabel('Amplitude')
plt.show()