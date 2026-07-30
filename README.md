# Foster's Inverse for SSS
Implementation of the MEG data preprocessing method derived and validated in [Noise Optimization of Basic Signal Component Extraction for Cryogenic and On-Scalp Magnetoencephalography (MEG)](https://www.biorxiv.org/content/10.64898/2026.07.21.739883v1), McPherson et al. July 2026

Adapting Signal Space Separation (SSS) to account for the impacts of sensor noise by weighting the inverse of the SSS matrix, resulting in a stable and accurate estimate of the multipole moments used to reconstruct the internal OPM-MEG data

## Parameters
Input `raw`
* `mne.raw` structure
* full raw MEG data file (ex. `.fif` format) from recording
* Type of sensor noise covariance to use: Empirical through `mne.compute_raw_covariance` or novel method using OTP to isolate sensor noise `mne.preprocessing.oversampled_temporal_projection`

Output `fos_raw`
* `mne.raw` structure
* raw strucutre with the MEG data updated with the Fosters Inverse preprocessed data
* `raw.info` structure updated to indicate some type of Maxwell Filtering/SSS preprocessing has occured
* Channels marked "bad" are dropped
