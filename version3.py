import os
import mne
import snirf
import numpy as np
from mne.io import read_raw_nirx, read_raw_snirf
from mne.preprocessing.nirs import beer_lambert_law, optical_density
from numpy.testing import assert_allclose
from mne_nirs.io import write_raw_snirf
import matplotlib.pyplot as plt

# Specify the path to the SNIRF file
snirf_file_path = "Desktop/KernelTest"

# Read the SNIRF file
raw = read_raw_snirf(snirf_file_path, verbose=True, preload=True)

events, event_id = mne.events_from_annotations(raw)

# Print original event IDs and their descriptions
print("Original Event ID mapping:")
print(event_id)

# Define the merging rules
# Merge events 1, 2, 3 into event 24
events = mne.merge_events(events, [1, 2, 3, 4, 5, 6], 24)

# Merge events 7, 8, 9 into event 25
events = mne.merge_events(events, [7, 8, 9, 10, 11, 12], 25)

# Print updated event IDs and their descriptions
unique_event_ids = np.unique(events[:, 2])
print(f"Unique event IDs after merging: {unique_event_ids}")

# Update the event_id mapping to reflect the new IDs
updated_event_id = {24: "Checkerboard", 25: "RedCross"}

# Print the updated event_id mapping
print("Updated Event ID mapping:")
print(updated_event_id)

events_dictionary = {
    "checkerboard": 24,  # Example ID for checkerboard
    "redcross": 25       # Example ID for redcross
}

#fig = mne.viz.plot_events(
#    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_dictionary
#)

#mne.viz.plot_events(events, sfreq=None, first_samp=0, color=None, event_id=events_dictionary, axes=None, equal_spacing=True, show=True, on_missing='raise', verbose=None)

def abs_func(raw):
    return np.abs(raw)

picks = mne.pick_types(raw.info, meg=False, fnirs=True)
raw.apply_function(abs_func, picks=picks)
raw.plot(
    n_channels=len(raw.ch_names), duration=500, show_scrollbars=False
)
#plt.show()


raw_od = optical_density(raw)
write_raw_snirf(raw_od, "test_raw_od.snirf")

result = snirf.validateSnirf("test_raw_od.snirf")
assert result.is_valid()
result.display()