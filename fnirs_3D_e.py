# ===============================
# AUTHOR: Andras Balint, andras.balint@unibe.ch / stefan.weder@insel.ch
# CREATE DATE: 23.08.2023
# PURPOSE: Contains functions for 3D scans and unique optode locations
# SPECIAL NOTES: Project: "How patterns of Brain Activation Predict Speech Understanding"
# ===============================
# Change History:
#   - 18.10.2023: - first commit to github  
# ===============================

import numpy as np
import mne
from scipy.spatial.transform import Rotation as R
import re

def dist_2D(P1, P2):
    # returns the distance of 2 coordinates (x,y)
    return np.sqrt(pow(P1[0]-P2[0],2) + pow(P1[1]-P2[1],2)) 

def dist_2D_pow(P1, P2):
    # returns the power of the distance of 2 coordinates (x,y)
    return pow(dist_2D(P1, P2),2)

def dist_3D(P1, P2):
    # returns the distance of 2 coordinates (x,y,z)
    return np.sqrt(pow(P1[0]-P2[0],2) + pow(P1[1]-P2[1],2) + pow(P1[2]-P2[2],2)) 

def MidPoint_3D(arr1, arr2):
    # returns the midpoint of 2 coordinates (x,y,z)
    return np.array([(arr1[0]+arr2[0])/2,(arr1[1]+arr2[1])/2,(arr1[2]+arr2[2])/2])


def update_coordinates(coords,labels):

    """ Returns (lpa, nas, rpa) coordinates from updated registered/template arrays

    ((lpa, nas, rpa) used in separate variables during processing)

    Args:
        coords (np.array): x-y-z coordinates
        labels (np.array): str optode labels
    Returns:
        (lpa, nas, rpa) (tuple): updated coordinates after fitment
    """  

    lpa = coords[np.where(labels == 'lpa')[0][0]]
    nas = coords[np.where(labels == 'nas')[0][0]]
    rpa = coords[np.where(labels == 'rpa')[0][0]]
    return (lpa, nas, rpa)


def GetStandardEEGMontage():

    """ Returns standard_1005 EEG positions as a template
    Args:
    Returns:
        eeg_elecpos (np.array): x-y-z coordinates of (template)
        eeg_labels (np.array): str optode labels (template)
    """

    eeg_montage = mne.channels.make_standard_montage('standard_1005')
    eeg_elecpos = np.zeros((len(eeg_montage.dig),3))
    eeg_labels = []
    eeg_labels.append('lpa')
    eeg_labels.append('nas')
    eeg_labels.append('rpa')
    for i in range(len(eeg_montage.dig)):
        eeg_elecpos[i] = eeg_montage.dig[i]['r']
        if i>2: # first ones are the lpa, rpa, nas
            eeg_labels.append(eeg_montage.ch_names[i-3])
    eeg_labels = np.array(eeg_labels)

    # remove duplicates
    vec = (eeg_labels == "T3") 
    vec = vec | (eeg_labels == "T5")
    vec = vec | (eeg_labels == "T6")
    vec = vec | (eeg_labels == "T4")
    eeg_labels = eeg_labels[~vec]
    eeg_elecpos = eeg_elecpos[~vec]

    return eeg_elecpos, eeg_labels


def ImportPPLocations(c_sub):
    
    """ reads .pp file
    Args:
        c_sub (str): path to .pp file
    Returns:
        mod_elecpos (np.array): x-y-z coordinates (registered)
        mod_labels (np.array): str optode labels (registerd)
    """

    f = open(c_sub, "r")
    lines = f.readlines()

    mod_elecpos = []
    mod_labels = []

    for line in lines:
        if 'point' in line:
            x = float(re.search('x="(.*?)"', line).group(1))
            y = float(re.search('y="(.*?)"', line).group(1))
            z = float(re.search('z="(.*?)"', line).group(1))
            label = re.search('name="(.*?)"', line).group(1)
            if label.startswith("T"): # Shimadzu specific
                label = label.replace("T", "S")
            if label.startswith("R"):
                label = label.replace("R", "D")
            mod_elecpos.append([x,y,z])
            mod_labels.append(label)    
    f.close()

    return np.array(mod_elecpos), np.array(mod_labels)


def fit_XYZ(eeg_elecpos, eeg_labels, mod_elecpos, mod_labels):

    """ One fitment in the iterative fitment process
    Args:
        eeg_elecpos (np.array): x-y-z coordinates of (template)
        eeg_labels (np.array): str optode labels (template)
        mod_elecpos (np.array): x-y-z coordinates (registered)
        mod_labels (np.array): str optode labels (registerd)
    Returns:
        mod_elecpos (np.array): updated coordinates (registered)
    """

    def to_arccos(p1, p2, p3):
        # to avoid numeric error
        c_val = (dist_2D_pow(p1,p2)+dist_2D_pow(p1,p3)-dist_2D_pow(p2,p3)) / (2*dist_2D(p1,p2)*dist_2D(p1,p3))
        if (c_val < -1) or (c_val > 1):
            return np.round(c_val)
        else:
            return c_val

    # turn over x-axis, to put "nas" on the y-axis (mod)
    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)
    p1 = [0,0]
    p2 = mod_nas[[1,2]]
    p3 = eeg_nas[[1,2]]
    rotations_rad = np.arccos(to_arccos(p1,p2,p3))
    rotations_rad = -rotations_rad # need minus, because it is the counter-clockwise direction

    rotation_axis = np.array([1,0,0])
    rotation_vector = rotations_rad * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    mod_elecpos = rotation.apply(mod_elecpos)

    # turn over y-axis, to put "lpa, rpa" on the x-axis (mod)
    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)
    p1 = [0,0]
    p2 = eeg_lpa[[0,2]]
    p3 = mod_lpa[[0,2]]
    rotations_rad = np.arccos(to_arccos(p1,p2,p3))
    p1 = [0,0]
    p2 = eeg_rpa[[0,2]]
    p3 = mod_rpa[[0,2]]
    rotations_rad += np.arccos(to_arccos(p1,p2,p3))
    rotations_rad = -(rotations_rad/2) # need minus, because it is the counter-clockwise direction
    rotation_axis = np.array([0,1,0])
    rotation_vector = rotations_rad * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    mod_elecpos = rotation.apply(mod_elecpos)

    # turn over z-axis, to put "nas, rpa, lpa" on the x-axis (mod)
    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)
    p1 = [0,0]
    p2 = eeg_nas[[0,1]]
    p3 = mod_nas[[0,1]]
    rotations_rad = np.arccos(to_arccos(p1,p2,p3))

    p1 = [0,0]
    p2 = eeg_lpa[[0,1]]
    p3 = mod_lpa[[0,1]]
    rotations_rad += np.arccos(to_arccos(p1,p2,p3))

    p1 = [0,0]
    p2 = eeg_rpa[[0,1]]
    p3 = mod_rpa[[0,1]]
    rotations_rad += np.arccos(to_arccos(p1,p2,p3))
    rotations_rad = -(rotations_rad/3) # need minus, because it is the counter-clockwise direction
    rotation_axis = np.array([0,0,1])
    rotation_vector = rotations_rad * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    mod_elecpos = rotation.apply(mod_elecpos)

    return mod_elecpos


def CheckMislabel(mod_elecpos, mod_labels, chs, default_d = 30, tolerance=3):

    """ Checks if the channels distances are within tolerance
    Args:
        mod_elecpos (np.array): x-y-z coordinates
        mod_labels (np.array): str optode labels
        chs (list): list of channels to check
        default_d (int): expected distance between channels in mm (=30mm)
        tolerance (int): tolerance for inter-optode distance in mm
    Returns:
        
    """
    MPs = []
    for c_ch_name in chs:
        c_S = c_ch_name.split('_')[0]
        c_D = c_ch_name.split('_')[1]

        # if channel exists
        if (np.sum(mod_labels == c_S) > 0) & (np.sum(mod_labels == c_D) > 0):
            c_S_coord = np.squeeze(mod_elecpos[mod_labels == c_S])
            c_D_coord = np.squeeze(mod_elecpos[mod_labels == c_D])

            c_MP = np.squeeze(dist_3D(c_S_coord, c_D_coord))*1000
            c_MP_passed = ((default_d - tolerance) < c_MP) & ((default_d + tolerance) > c_MP)

            assert c_MP_passed == True, f"CH: {c_ch_name} registered distance: {np.round(c_MP,2)} expected distance: {default_d}" 
            
            MPs.append(c_MP)
        else:
            print('No 3D registration for channel: ' + str(c_ch_name))
    
    return MPs


def CoordinatesPipeline(mod_elecpos, mod_labels):

    """ Main pipeline to fit the coordinates to the template
    Args:
        mod_elecpos (np.array): x-y-z coordinates (registered)
        mod_labels (np.array): str optode labels (registerd)
    Returns:
        mod_elecpos (np.array): updated coordinates (registered)
    """

    # init
    eeg_elecpos, eeg_labels = GetStandardEEGMontage()
    eeg_elecpos_0 = eeg_elecpos.copy()
    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)

    # calculate middlepoint
    mod_o = MidPoint_3D(mod_lpa, mod_rpa)
    eeg_o = MidPoint_3D(eeg_lpa, eeg_rpa)

    # translation to have the same middlepoint
    m_translation_shift = mod_o-eeg_o
    mod_elecpos = mod_elecpos-m_translation_shift

    # translation to have middlepoint as origo (need to do it back later)
    mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)
    mod_translation_origo = MidPoint_3D(mod_lpa,mod_rpa)
    mod_elecpos = mod_elecpos - mod_translation_origo

    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    eeg_translation_origo = MidPoint_3D(eeg_lpa,eeg_rpa)
    eeg_elecpos = eeg_elecpos - eeg_translation_origo

    # turn over x-axis, to put "nas" on the y-axis (EEG)
    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    rotations_rad = np.arccos(eeg_nas[1]/dist_2D([0,0], eeg_nas[[1,2]])) # !!! this returns in radius
    rotations_rad = -rotations_rad # need minus, because it is the counter-clockwise direction
    rotation_axis = np.array([1,0,0])
    rotation_vector = rotations_rad * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    eeg_elecpos = rotation.apply(eeg_elecpos)

    # scaling (MOD)
    eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
    mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)
    scaling_factor = dist_3D([0,0,0], eeg_lpa)/dist_3D([0,0,0], mod_lpa)
    scaling_factor += dist_3D([0,0,0], eeg_rpa)/dist_3D([0,0,0], mod_lpa)
    scaling_factor += dist_3D([0,0,0], eeg_nas)/dist_3D([0,0,0], mod_nas)
    scaling_factor = scaling_factor/3
    mod_elecpos = mod_elecpos*scaling_factor

    # transformations to reverse when finished
    reverse_x_rotation = rotations_rad # the rotation_axis = np.array([1,0,0])

    # fit to XYZ-axis iteratively
    mod_elecpos_1 = mod_elecpos.copy()
    abs_error = []
    for i in range(50):
        mod_elecpos_1 = fit_XYZ(eeg_elecpos, eeg_labels, mod_elecpos_1, mod_labels)
        eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
        mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos_1, mod_labels)
        abs_error.append(dist_3D(eeg_nas, mod_nas)+dist_3D(eeg_lpa, mod_lpa)+dist_3D(eeg_rpa, mod_rpa))
    abs_error = [999 if np.isnan(x) else x for x in abs_error] # if fails, it put's 999 into the list..
    for i in range(np.argmin(abs_error)+1):
        mod_elecpos = fit_XYZ(eeg_elecpos, eeg_labels, mod_elecpos, mod_labels)
        eeg_lpa, eeg_nas, eeg_rpa = update_coordinates(eeg_elecpos, eeg_labels)
        mod_lpa, mod_nas, mod_rpa = update_coordinates(mod_elecpos, mod_labels)
        abs_error_final = dist_3D(eeg_nas, mod_nas)+dist_3D(eeg_lpa, mod_lpa)+dist_3D(eeg_rpa, mod_rpa)
    print("Iterative fitment ({:.0f}x) reached absolute error: {:.4f}".format(np.argmin(abs_error)+1,abs_error_final))

    # Moving electrodes inward
    distsum = 0
    distidx = 0
    distvec = np.zeros(len(mod_labels),dtype=bool)

    for i in range(len(mod_labels)):
        if mod_labels[i] not in ['lpa', 'nas', 'rpa']:
            dists = np.zeros(eeg_elecpos.shape[0])
            for j in range(len(dists)):
                dists[j] = dist_3D(eeg_elecpos[j], mod_elecpos[i])
            p_eeg = eeg_elecpos[np.argmin(dists)]
            p_mod = mod_elecpos[i]
            distsum += dist_3D([0,0,0], p_eeg) / dist_3D([0,0,0], p_mod)
            distidx += 1
            distvec[i] = 1

    tmp = mod_elecpos[distvec] * (distsum/distidx)
    mod_elecpos[distvec] = tmp

    # transformations to reverse: Rotation
    rotations_rad = -reverse_x_rotation # perform in opposite direction
    rotation_axis = np.array([1,0,0])
    rotation_vector = rotations_rad * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    mod_elecpos = rotation.apply(mod_elecpos)
    # transformations to reverse: Translation
    mod_elecpos = mod_elecpos + eeg_translation_origo # perform in opposite direction

    # just to check if original coordinate system is back
    eeg_elecpos = rotation.apply(eeg_elecpos)
    eeg_elecpos = eeg_elecpos + eeg_translation_origo # perform in opposite direction
    assert (eeg_elecpos_0 - eeg_elecpos).sum() < 1e-15
    
    return mod_elecpos
