import os
import mne
import snirf
import numpy as np
import pandas as pd
from mne.io import read_raw_snirf
from mne.preprocessing.nirs import beer_lambert_law, optical_density
from mne_nirs.io import write_raw_snirf
from numpy.testing import assert_allclose
import h5py

mne.viz.set_browser_backend('matplotlib') #qt for interactive visualization

# Specify the path to the SNIRF file
snirf_file_path = "Kernel_S001_2e8a8eb_5.snirf"

# Read the SNIRF file
raw_intensity = read_raw_snirf(snirf_file_path, verbose=True, preload=True).load_data()
raw_intensity.apply_function(lambda x: x * 1e-6)
raw_intensity.plot(n_channels=20, duration=30, show_scrollbars=False)
fig.savefig('raw_plot_20ch.png', bbox_inches='tight')
#raw_intensity.plot(n_channels=8368, duration=30, show_scrollbars=False)
#fig.savefig('raw_plot_all_ch.png', bbox_inches='tight')

def inspect_stim1_data(file_path):
    stim1_contents = {}
    with h5py.File(file_path, 'r') as f:
        if 'nirs/stim1' in f:
            for key, dataset in f['nirs/stim1'].items():
                try:
                    stim1_contents[key] = dataset[()]
                except Exception as e:
                    stim1_contents[key] = str(e)
    return stim1_contents

# Provide the path to your uploaded original SNIRF file
stim1_contents = inspect_stim1_data(snirf_file_path)
stim1_contents



################

# Extract data and channel names
data = raw_intensity.get_data()
channel_names = raw_intensity.ch_names

# Create a DataFrame with the data
df = pd.DataFrame(data.T, columns=channel_names)

# Print the number and percentage of NaNs
num_nans = df.isna().sum().sum()
total_data_points = df.size
percentage_nans = (num_nans / total_data_points) * 100

print(f"Number of NaNs: {num_nans}")
print(f"Percentage of NaNs: {percentage_nans:.2f}%")

# Mark channels with NaNs as bad
for i, channel in enumerate(channel_names):
    if np.isnan(data[i, :]).any():
        raw_intensity.info['bads'].append(channel)

# Interpolate bad channels
raw_intensity.interpolate_bads(reset_bads=True)

# Extract the interpolated data
data_interpolated = raw_intensity.get_data()

# Check if there are any remaining NaNs after interpolation
num_nans_after = np.isnan(data_interpolated).sum()
print(f"Number of NaNs after interpolation: {num_nans_after}")

# If there are still NaNs, use fallback methods
if num_nans_after > 0:
    print("Using fallback methods to handle remaining NaNs.")
    # Convert data to DataFrame for easier manipulation
    df_interpolated = pd.DataFrame(data_interpolated.T, columns=channel_names)
    # Forward fill and backward fill as fallback
    df_interpolated = df_interpolated.ffill().bfill()
    # Check the number of NaNs after fallback methods
    num_nans_final = df_interpolated.isna().sum().sum()
    print(f"Number of NaNs after forward/backward fill: {num_nans_final}")
    # Ensure no NaNs remain
    assert num_nans_final == 0, "There are still NaNs remaining after fallback methods."
    # Convert back to NumPy array
    data_interpolated = df_interpolated.T.values
    # Replace the data in the Raw object with the fully interpolated data
    raw_intensity._data = data_interpolated

# Now you can use `raw_intensity` for further processing
print(raw_intensity)

# Write the interpolated data back to disk in the SNIRF format
write_raw_snirf(raw_intensity, "test_raw_interpolated.snirf")

# Read back SNIRF file
snirf_intensity = read_raw_snirf("test_raw_interpolated.snirf")

# Compare files
assert_allclose(raw_intensity.get_data(), snirf_intensity.get_data())

# Plot the data
snirf_intensity.plot(n_channels=30, duration=300, show_scrollbars=False)

# Validate SNIRF File
result = snirf.validateSnirf("test_raw_interpolated.snirf")
assert result.is_valid()
result.display()

# Optical Density
raw_od = optical_density(raw_intensity)
write_raw_snirf(raw_od, "test_raw_od.snirf")

result = snirf.validateSnirf("test_raw_od.snirf")
assert result.is_valid()
result.display()

# Haemoglobin
raw_hb = beer_lambert_law(raw_od)
write_raw_snirf(raw_hb, "test_raw_hb.snirf")

result = snirf.validateSnirf("test_raw_hb.snirf")
assert result.is_valid()
result.display()
