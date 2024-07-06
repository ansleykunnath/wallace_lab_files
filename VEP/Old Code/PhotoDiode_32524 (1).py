import mne # import the MNE python package for EEG analysis 
import numpy as np # numpy for numerical computing 
from mne.preprocessing import (ICA) # independent component analysis 
from autoreject import AutoReject # for automatic artifact rejection 
import matplotlib # data visualization 
matplotlib.use("TkAgg") # backend for matplotlib to TkAgg

eeg_path = "C://Users//neuro//Documents//Python Scripts//VEP//VEP Data"  # You will need to change this location # define path 
file_name = "AJK-03-01-24" # file name of eeg data 
file_eeg = eeg_path + file_name + ".eeg" # add the file types to the end 
file_vhdr = eeg_path + file_name + ".vhdr"
file_vmrk = eeg_path + file_name + ".vmrk"

raw = mne.io.read_raw_brainvision(file_vhdr) # read in raw eeg data from the brainvision file 
#drop_channels = ['BIP1','BIP2','EOG','TEMP1','ACC1','ACC2','ACC3']
raw = raw.drop_channels(drop_channels) # plot the raw eeg data after dropping any unwanted channels from the eeg data 
raw.plot()
events_from_annot, event_dict = mne.events_from_annotations(raw) # extract events and event dictionary from the annotations in the eeg data 
del event_dict['Stimulus/s5'] # remove any unwanted events from the event dictionary 

highpass = 1 # define highpass filter frequency 
lowpass = 20 # low pass filter frequency 
notch = 60 # define notch filter frequency 

raw_filtered = raw.load_data().filter(highpass, lowpass).notch_filter(np.arange(notch, (notch * 3), notch)) # copy filtered eeg data and apply an average reference montage 

#raw_filtered = raw.resample(resample).filter(highpass, lowpass).notch_filter(np.arange(notch, (notch * 3), notch))

eeg_1020 = raw_filtered.copy().set_eeg_reference(ref_channels = 'average') # ref_channels='['Fz']' 
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') # define the 10-20 electrode montage 
eeg_1020 = eeg_1020.set_montage(ten_twenty_montage, on_missing = 'ignore')
del raw, raw_filtered, ten_twenty_montage # apply to data 
eeg_1020.info['bads'] = [] # delete the unnecessary variables and define any bad channels in the eeg data 
picks = mne.pick_types(eeg_1020.info, meg=False, eeg=True, stim=False, eog=False, include=[], exclude=[]) # define which eeg channels to be used in the analysis 

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
                    event_repeated='drop') # creating epochs from the data 

n_interpolates = np.array([1, 4, 32]) # define an array of values for the number of interpolation steps in AutoReject 
consensus_percs = np.linspace(0, 1.0, 11) # define the array of values 
ar = AutoReject(n_interpolates, # create instance of autoreject class with specified parameters
                consensus_percs,
                picks=picks,
                thresh_method='random_search',
                random_state=42)    #random n state
epochs_ar = ar.fit_transform(epochs) # save cleaned epochs to a new variable 

ica = ICA(n_components = 16, max_iter = 'auto', random_state = 123) # create an instance of the ica class with specified parameters 
ica.fit(epochs_ar)

ica_z_thresh = 1.96 # fit the ica to the cleaned epochs ; define threshold for detecting eye blink 
epochs_clean = epochs_ar.copy() # copy the cleaned epochs to a new variable 
eog_indices, eog_scores = ica.find_bads_eog(epochs_clean,
                                            ch_name=['Fp1', 'F6'],
                                            threshold=ica_z_thresh) # detect eye blink artifacts in the epochs using the ICA and save indices and scores of the bad components 
ica.exclude = eog_indices # exclude bad components from the ica 
print(eog_indices) # print indices of the excluded components 
ica.apply(epochs_clean) # apply ica to the cleaned epochs to remove eye blink artifacts 
epochs_final = epochs_clean.copy() # save the final cleaned epochs to a new variable 
del eeg_1020, epochs, epochs_ar, eog_indices, eog_scores, drop_channels # delete unnecessary variables to save memory 

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

baseline_tmin, baseline_tmax = -0.05, 0 # define the baseline window for averaging the epochs 
baseline = (baseline_tmin, baseline_tmax) # define baseline window as a tuple 

VEP = epochs_final['Stimulus/s1'].apply_baseline(baseline).average() # extract and average epochs for first stimulus condition to create a vep 

################### new as of 3/25/24 for code to find peak amplitude of the VEP waveform within the time window after the line that defines the baseline window 
# Find the time indices corresponding to the time window of interest

tmin, tmax = 0.08, 0.12 # how long is the stimulus? When does it blink? 
time_inds = np.where((VEP.times >= tmin) & (VEP.times <= tmax))[0] #for defining time ranges 

# Find the peak amplitude and time of the VEP waveform within the time window
peak_amp = VEP.data[0, time_inds].max() # find the max 
peak_time = VEP.times[time_inds][VEP.data[0, time_inds].argmax()] 

# Print the peak amplitude and time
print("Peak amplitude: ", peak_amp)
print("Peak time: ", peak_time)

# Find the time indices corresponding to the time window of interest
tmin, tmax = 0.08, 0.12
time_inds = np.where((VEP.times >= tmin) & (VEP.times <= tmax))[0]

# Find the indices and values of the lowest amplitudes in the time window
low_amp_inds = np.where(VEP.data[0, time_inds] == VEP.data[0, time_inds].min())[0]
low_amp_vals = VEP.data[0, time_inds][low_amp_inds]

# Find the times corresponding to the lowest amplitudes- where the goal of finding the lowest amplitudes is to find the difference between flashing screen and not 
low_amp_times = VEP.times[time_inds][low_amp_inds]

# Print the lowest amplitude values and times
print("Lowest amplitude values: ", low_amp_vals)
print("Times of lowest amplitudes: ", low_amp_times)

#############################

VEP_2 = epochs_final['Stimulus/s3'].apply_baseline(baseline).average() # extract and average the epochs for the second stimulus condition to create a vep 
blank = epochs_final['Stimulus/s2'].apply_baseline(baseline).average() # for blank condition to create a vep 
#VEP = epochs_final['Stimulus/s1'].average()
#VEP_2 = epochs_final['Stimulus/s3'].average()
#blank = epochs_final['Stimulus/s2'].average()

#fig = mne.viz.plot_compare_evokeds(VEP, picks='Oz', show=False)
#fig[0].savefig("VEP Data/VEP_Oz")

fig = mne.viz.plot_compare_evokeds(VEP, picks=picks, combine="mean", show=False) # for plotting 
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
fig[0].savefig("VEP Data/VEP_Occipital") # for plotting the vep for the first stimulus condition using MNE python 

#fig = mne.viz.plot_compare_evokeds(VEP_shift, picks=picks, combine="mean", show=False, time_unit="ms")

fig = mne.viz.plot_compare_evokeds(VEP, picks=['POz','O1','O2'], combine="mean", show=True, time_unit="ms") # using the mne python 
fig[0].savefig("VEP Data/VEP_Occipital") 

fig = mne.viz.plot_compare_evokeds(VEP, picks=['POz'], show=True, time_unit="ms")
fig[0].savefig("VEP Data/VEP_POz") 

#fig = mne.viz.plot_compare_evokeds(dict(Checkboard=VEP, Blank=blank), colors=dict(Checkboard="orange", Blank="black"), picks=['O1','O2','Oz','POz','PO4','PO6','PO8','PO3','PO5','PO7'], time_unit="ms", combine="mean")
#fig[0].savefig("VEP Data/Compare_Stimuli")
