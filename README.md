# Foster's Inverse for SSS
Adapting Signal Space Separation (SSS) to account for the impacts of sensor noise by weighting the inverse of the SSS matrix, resulting in a stable and accurate estimate of the multipole moments used to reconstruct the internal OPM-MEG data

## Parameters
Input `raw`
* `mne.raw` structure
* full raw MEG data file (ex. `.fif` format) from recording with `raw.info["bads"]` properly marked

Output `fos_raw`
* `mne.raw` structure
* raw strucutre with the MEG data updated with the Fosters Inverse preprocessed data
* `raw.info` structure updated to indicate some type of Maxwell Filtering/SSS preprocessing has occured
* Channels marked "bad" are dropped
