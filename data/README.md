# Data directory

Place or symlink your MEG recordings here. Data files are git-ignored.

Example layout:

```text
data/
  subject-01/
    recording_raw.fif
    empty_room_raw.fif
    clean_raw.fif
    noise-cov.fif
```

Run the comparison with paths relative to the repository root:

```bash
python main.py --raw data/subject-01/recording_raw.fif \
  --noise-raw data/subject-01/empty_room_raw.fif \
  --noise-tmin 0 --noise-tmax 60 \
  --out results/subject-01
```

The input must contain MEG channels. Supported extensions are listed in the
main README. To add another MNE reader, edit `foster_sss/data.py`.
