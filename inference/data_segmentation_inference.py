import mne
import numpy as np
from subject_segments_all_cleaned import subject_segments
import glob
import re
import gc
from pathlib import Path


def epoch_data(eeg_segment, window_size_time, overlap=0.0):
    sfreq = eeg_segment.info["sfreq"]
    win_samples = int(window_size_time * sfreq)

    # hop = win_len × (1 − overlap)  →  50 %  ⇒ hop = win_len / 2
    hop_samples = int(win_samples * (1 - overlap))

    starts = np.arange(
        0, len(eeg_segment) - win_samples + 1,  # inclusive last start
        hop_samples
    )

    events = np.vstack([
        starts + eeg_segment.first_samp,  # sample index in Raw
        np.zeros(len(starts), dtype=int),  # dummy event IDs
        np.ones(len(starts), dtype=int)
    ]).T

    epochs = mne.Epochs(
        eeg_segment,
        events,
        tmin=0,
        tmax=window_size_time - (1 / sfreq),
        baseline=(0, 0),
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose=False,
    )
    return epochs.get_data()


def eeg_read(vhdr_file, periods, overlap):
    window_size_time = periods * 0.125  # n × TR (more context)

    # --- Read full raw recording (EEG + ECG)
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=False)

    # Extract subject ID from filename
    match = re.search(r'(\d{3})_', vhdr_file)
    if match:
        subject_id = int(match.group(1))
    print(f"\nProcessing Subject {subject_id}")
    SUBJECT_DIR = OUTPUT_DIR / str(subject_id)
    SUBJECT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Detect and select all EEG channels and the ECG channel ---
    all_eeg_chs = raw.info["ch_names"][0:24]  # first 24 EEG channels
    ecg_ch = raw.info["ch_names"][-1]          # ECG is usually last
    all_chs = all_eeg_chs + [ecg_ch]           # combine EEG + ECG

    print(f"Detected {len(all_eeg_chs)} EEG channels and 1 ECG channel ({ecg_ch}).")

    # --- Apply selections and preprocessing ---
    raw.pick(all_chs)
    raw.resample(512.)
    raw.set_annotations(None)
    raw = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)

    contaminated_segments = []

    # --- Contaminated array writing ---
    print("#### Writing Contaminated Segments ... ####")
    for contaminated_time_slot in subject_segments[subject_id]["contaminated"]:
        print(contaminated_time_slot)
        contaminated_seg = raw.copy().crop(
            tmin=contaminated_time_slot[0], tmax=contaminated_time_slot[1]
        )
        contaminated_data = epoch_data(contaminated_seg, window_size_time, overlap)
        contaminated_segments.append(contaminated_data)

        del contaminated_seg, contaminated_data
        gc.collect()
        break
        #!! remove `break` so it writes all segments for the subject

    # --- Concatenate all epochs along axis 0 (n_epochs) ---
    contaminated_segments = np.concatenate(contaminated_segments, axis=0)

    print(f"Saving contaminated.npy with shape {contaminated_segments.shape} "
          f"(n_epochs, {len(all_chs)}, T)")

    out_path = SUBJECT_DIR / "contaminated.npy"
    np.save(out_path, contaminated_segments)

    del contaminated_segments
    gc.collect()


def main():
    periods = 10  # number of TRs per window
    overlap = 0.0

    data_paths = glob.glob("current_study_data_raw/H097/H097_scan.vhdr")
    print("---- Subject Data Segmentation ----")
    for subject_vhdr in data_paths:
        eeg_read(subject_vhdr, periods, overlap)


if __name__ == "__main__":
    OUTPUT_DIR = Path("data_segmented_97_10_TR_all_channels_with_ecg_interference")
    main()
