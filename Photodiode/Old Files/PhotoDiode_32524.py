import mne
import numpy as np
from mne.preprocessing import (ICA)
from autoreject import AutoReject
import matplotlib
matplotlib.use("TkAgg")

eeg_path = "C://Users//neuro//Documents//Python Scripts//VEP//VEP Data"  # You will need to change this location
file_name = "AJK-03-01-24"
file_eeg = eeg_path + file_name + ".eeg"
file_vhdr = eeg_path + file_name + ".vhdr"
file_vmrk = eeg_path + file_name + ".vmrk"

raw = mne.io.read_raw_brainvision(file_vhdr)
#drop_channels = ['BIP1','BIP2','EOG','TEMP1','ACC1','ACC2','ACC3']
raw = raw.drop_channels(drop_channels)
raw.plot()
events_from_annot, event_dict = mne.events_from_annotations(raw)
del event_dict['Stimulus/s5']

highpass = 1
lowpass = 20
notch = 60

raw_filtered = raw.load_data().filter(highpass, lowpass).notch_filter(np.arange(notch, (notch * 3), notch))
#raw_filtered = raw.resample(resample).filter(highpass, lowpass).notch_filter(np.arange(notch, (notch * 3), notch))

eeg_1020 = raw_filtered.copy().set_eeg_reference(ref_channels = 'average') # ref_channels='['Fz']'
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
eeg_1020 = eeg_1020.set_montage(ten_twenty_montage, on_missing = 'ignore')
del raw, raw_filtered, ten_twenty_montage
eeg_1020.info['bads'] = []
picks = mne.pick_types(eeg_1020.info, meg=False, eeg=True, stim=False, eog=False, include=[], exclude=[])

epochs = mne.Epochs(eeg_1020,
                    events=events_from_annot,
                    event_id=event_dict,
                    tmin=-0.05,
                    tmax=0.3,   #duration of stimulus or response
                    baseline=None,
                    reject=None,
                    verbose=False,
                    preload=True,
                    detrend=None,
                    event_repeated='drop')

n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)
ar = AutoReject(n_interpolates,
                consensus_percs,
                picks=picks,
                thresh_method='random_search',
                random_state=42)    #random n state
epochs_ar = ar.fit_transform(epochs)

ica = ICA(n_components = 16, max_iter = 'auto', random_state = 123)
ica.fit(epochs_ar)

ica_z_thresh = 1.96
epochs_clean = epochs_ar.copy()
eog_indices, eog_scores = ica.find_bads_eog(epochs_clean,
                                            ch_name=['Fp1', 'F6'],
                                            threshold=ica_z_thresh)
ica.exclude = eog_indices
print(eog_indices)
ica.apply(epochs_clean)
epochs_final = epochs_clean.copy()
del eeg_1020, epochs, epochs_ar, eog_indices, eog_scores, drop_channels

########################################
#### Manual ICA Analysis #####
##ica.plot_sources(epochs)
#ica.plot_components()
##ica.plot_properties(epochs)
#exclude = [0,1,2]    # select based on ICA abnl. OPTIONAL.
#epochs_clean = epochs.copy()
#ica.exclude = exclude
#ica.apply(epochs_clean)
#epochs_clean.plot(n_channels = len(epochs_clean))
#epochs_final = epochs_clean.copy()

## if no ICAs:
#epochs_final = epochs_clean.copy()
#del eeg_1020, epochs, epochs_clean
########################################

baseline_tmin, baseline_tmax = -0.05, 0
baseline = (baseline_tmin, baseline_tmax)

VEP = epochs_final['Stimulus/s1'].apply_baseline(baseline).average()
VEP_2 = epochs_final['Stimulus/s3'].apply_baseline(baseline).average()
blank = epochs_final['Stimulus/s2'].apply_baseline(baseline).average()
#VEP = epochs_final['Stimulus/s1'].average()
#VEP_2 = epochs_final['Stimulus/s3'].average()
#blank = epochs_final['Stimulus/s2'].average()

#fig = mne.viz.plot_compare_evokeds(VEP, picks='Oz', show=False)
#fig[0].savefig("VEP Data/VEP_Oz")

fig = mne.viz.plot_compare_evokeds(VEP, picks=picks, combine="mean", show=False)
#fig[0].savefig("VEP Data/VEP_All")

#fig = mne.viz.plot_compare_evokeds(VEP, picks=['O1','O2','Oz','POz','PO3','PO4','PO5','PO6','PO7','PO8'], combine="mean", show=False, time_unit="ms")
#fig[0].savefig("VEP Data/VEP_Occipital1")

#fig = mne.viz.plot_compare_evokeds(VEP_shift, picks='Oz', show=False, time_unit="ms")
#fig[0].savefig("VEP Data/VEP_1_Oz")

#fig = mne.viz.plot_compare_evokeds(VEP_shift, picks='Oz', show=False, time_unit="ms")
#fig[0].savefig("VEP Data/VEP_All_Oz")

#fig = mne.viz.plot_compare_evokeds(VEP_shift, picks='POz', show=False, time_unit="ms")
#fig[0].savefig("VEP Data/VEP_All_POz")

#fig = mne.viz.plot_compare_evokeds(dict(Checkboard=VEP, Check_2=VEP_2, Blank=blank), colors=dict(Checkboard="orange", Check_2="red", Blank="black"), picks=['Oz','O1','O2','POz'], time_unit="ms", combine="mean")
#fig[0].savefig("VEP Data/Compare_Stimuli")

#epochs_final['Stimulus/s1'].plot(n_epochs=1, events=True, picks='Oz')

fig = mne.viz.plot_compare_evokeds(VEP, picks=['O1','O2','Oz','POz','PO4','PO6','PO8','PO3','PO5','PO7'], combine="mean", show=True, time_unit="ms")
fig[0].savefig("VEP Data/VEP_Occipital")

#fig = mne.viz.plot_compare_evokeds(VEP_shift, picks=picks, combine="mean", show=False, time_unit="ms")

fig = mne.viz.plot_compare_evokeds(VEP, picks=['POz','O1','O2'], combine="mean", show=True, time_unit="ms")
fig[0].savefig("VEP Data/VEP_Occipital")

fig = mne.viz.plot_compare_evokeds(VEP, picks=['POz'], show=True, time_unit="ms")
fig[0].savefig("VEP Data/VEP_POz")

#fig = mne.viz.plot_compare_evokeds(dict(Checkboard=VEP, Blank=blank), colors=dict(Checkboard="orange", Blank="black"), picks=['O1','O2','Oz','POz','PO4','PO6','PO8','PO3','PO5','PO7'], time_unit="ms", combine="mean")
#fig[0].savefig("VEP Data/Compare_Stimuli")
