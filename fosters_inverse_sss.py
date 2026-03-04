#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:40:55 2026

@author: alexandria mcpherson


Fosters Inverse with SSS accounts for the impacts of sensor noise by weighting the inverse
of the SSS matrix, resulting in a stable and accurate estimate of the multipole moments
used to reconstruct the internal MEG data

INPUTS
- "raw" structure loaded using MNE-Python mne.io.read_raw_fif

OUTPUTS
- denoised, preprocessed "raw" strucutre with indication of maxwell filtering added to info

"""

# --- Dependencies ------------------------------------------------------------
import os
import numpy as np
import mne
import mne.utils

# --- Functions ------------------------------------------------------------
def _do_inverse(raw,N):
    """
    Parameters
    ----------
    raw : mne.raw structure
        full raw meg file, ex. "fif", from recording with raw.info["bads"] indicated
    N : 2D square matrix, (number of sensors) X (number of sensors)
        Sensor noise covariance matrix, calculated using empircial covariance
        implemented in mne.compute_raw_covariance

    Returns
    -------
    data_fosters : 2D matrix, (number of sensors) X (time)
        Matrix containing data corresponding to each MEG channel over time after
        reconstruction with Fosters Inverse preprocessing
    """
    ## extract raw data matrix from MEG channels
    phi_0 = raw.get_data(picks='meg')
    ## calculate SSS matrix S and multiple moments with reccomended params
    [S, pS, reg_moments, n_use_in]=mne.preprocessing.compute_maxwell_basis(raw.info, origin=(0.,0.,0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
    
    ## setup Foster's Inverse- calculate Matrix B and vector b
    S = S[:, :n_use_in]
    XN = pS[:n_use_in,:] @ phi_0
    ## for full S
    # XN = pS @ phi_0
    alpha = np.transpose(XN)
    alpha_cov_norm = np.cov(XN)
    S_star = np.transpose(np.conj(S))
    first = np.linalg.pinv(S@alpha_cov_norm@S_star +N)
    B = alpha_cov_norm @ S_star @ first
    m_alpha = np.transpose(np.mean(alpha,0))
    b = m_alpha - B@S@m_alpha
    x_bar = np.zeros_like(XN)
    
    ## calculate Foster's Inverse estimate of multipole moments
    for i in range(0,np.shape(phi_0)[1]):
        x_bar[:,i]=B@phi_0[:,i] + b
    
    ## use new estimate to reconstruct internal data
    data_fosters = np.real(S[:, :n_use_in]@x_bar[:n_use_in,:])
    return data_fosters
    
def fosters_inverse(raw):
    """
    Parameters
    ----------
    raw : mne.raw structure
        full raw meg file, ex. "fif", from recording with raw.info["bads"] indicated
    
    Returns
    -------
    raw_fos : mne.raw structure
        raw strucutre with the MEG data updated with the Fosters Inverse 
        preprocessed data, raw.info structure updated to indicate some type of
        Maxwell Filtering/SSS preprocessing has occured. Channels marked "bad" 
        are dropped
    """
    ## calculate sensor noise covariance
    N = mne.compute_raw_covariance(raw,rank="info",method='empirical')["data"]
    ## drop bad channels 
    bads = raw.info["bads"]
    raw.drop_channels(bads)
    ## create data strcutre, indicates in "info" that some preprocessing akin to SSS has happened
    raw_fos = mne.preprocessing.maxwell_filter(raw, origin=(0.,0.,0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)  # just to get the info to indicate some Maxwell filtering was done etc.
    assert raw.info["bads"] == [] # double check bads were dropped
    
    ## Do foster's inverse!
    foster_sss_data= _do_inverse(raw, N)
    
    ## isolate MEG channels 
    meg_picks = mne.pick_types(raw.info, meg=True)
    ## put new Foster's inverse recon data into "raw" structure
    raw_fos._data[meg_picks] = foster_sss_data
    
    ## cleanup
    del foster_sss_data
    
    return raw_fos
    

# --- Example Main ------------------------------------------------------------
if __name__ == '__main__':
    ## define raw file directory and raw file names
    raw_dir ='~/sub-XM'
    raw_file = '~_raw.fif'
    
    ## load data and events
    raw = mne.io.read_raw_fif(os.path.join(raw_dir, raw_file),preload=False, allow_maxshield='no')
    trigger_chan = 'di2'
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    
    ## high and low - pass raw data
    freq_min = 0.1
    freq_max = 50
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)

    ## call Foster's inverse
    raw_fos = fosters_inverse(raw)
    
    ## calculate and plot evoked 
    tmin = -0.1  # start of each epoch (200ms before the trigger)
    tmax = 0.4  # end of each epoch (400ms after the trigger)
    epochs = mne.Epochs(raw_fos, events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    evoked = epochs.average()
    fig = evoked.plot_joint(times=[0.047,0.1], title="Low-Pass 50 Hz, Foster's Inverse")

    
    