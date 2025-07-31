import mne
import numpy as np
from subject_segments_all_cleaned import subject_segments
import glob
import re
import gc
from pathlib import Path

def epoch_data(eeg_segment, window_size_time, overlap=0.0):
    sfreq       = eeg_segment.info["sfreq"]
    win_samples = int(window_size_time * sfreq)

    # hop = win_len × (1 − overlap)  →  50 %  ⇒ hop = win_len / 2
    hop_samples = int(win_samples * (1 - overlap))

    starts = np.arange(0, len(eeg_segment) - win_samples + 1,   # inclusive last start
        hop_samples
    )

    events = np.vstack([
        starts + eeg_segment.first_samp,      # sample index in Raw
        np.zeros(len(starts), dtype=int),     # dummy event IDs
        np.ones (len(starts), dtype=int)
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
    
    window_size_time = periods * 0.125 # n * TR (more context)
    
    # read raw recording
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=False)

    match = re.search(r'(\d{3})_', vhdr_file)
    if match:
        subject_id = int(match.group(1))
    print(subject_id)
    SUBJECT_DIR = OUTPUT_DIR / str(subject_id)
    SUBJECT_DIR.mkdir(parents=True, exist_ok=True)

    chs = raw.info["ch_names"][0:24]

    raw.pick(chs)
    raw.resample(512.)
    raw.set_annotations(None)
    raw = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)

    clean_segments = []
    contaminated_segments = []

    # Clean Array Writing
    print('#### Writing Clean Segments ... ####')
    for clean_time_slot in subject_segments[subject_id]['clean']:
        seg = raw.copy().crop(tmin=clean_time_slot[0], tmax=clean_time_slot[1])
        data = epoch_data(seg, window_size_time, overlap)
        clean_segments.append(data)
        del seg, data
        gc.collect()

    # Concatenate all epochs along axis 0 (n_epochs)
    clean_segments = np.concatenate(clean_segments, axis=0)
    
    # Save subject's clean.npy
    out_path = SUBJECT_DIR / 'clean.npy'
    np.save(out_path, clean_segments)
    
    del clean_segments
    gc.collect()

    # contaminated Array Writing
    print('#### Writing Contaminated Segments ... ####')
    for contaminated_time_slot in subject_segments[subject_id]['contaminated']:
        contaminated_seg = raw.copy().crop(tmin=contaminated_time_slot[0], tmax=contaminated_time_slot[1])
        contaminated_data = epoch_data(contaminated_seg, window_size_time, overlap)
        contaminated_segments.append(contaminated_data)

        del contaminated_seg, contaminated_data
        gc.collect()

    # Concatenate all epochs along axis 0 (n_epochs)
    contaminated_segments = np.concatenate(contaminated_segments, axis=0)

    # Save subject's contaminated.npy
    out_path = SUBJECT_DIR / 'contaminated.npy'
    np.save(out_path, contaminated_segments)
       
    del contaminated_segments
    gc.collect()

def eeg_read_noise(periods, overlap):
    watermelon = mne.io.read_raw_brainvision('current_study_data_raw/watermelon2.0/watermelon2.0_scan.vhdr', preload=True)

    chs = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','FPz','Fz','Cz','Pz',
    'POz','Oz','FT9','FT10',"TP9'","TP10'",
    ]
    watermelon.pick(chs)
    watermelon.crop(tmin=1855.415, tmax=2813.099)
    watermelon.set_annotations(None)
    watermelon.resample(512.)
    
    # Remove slow drift
    watermelon_hp = watermelon.copy().filter(l_freq=0.5, h_freq=None, verbose=False)

    window_size_time = periods * 0.125 # 3 TR
    noise_data = epoch_data(watermelon_hp, window_size_time, overlap)

    print('#### Writing Noise Segments ... ####')
    out_path = OUTPUT_DIR / 'watermelon.npy'
    np.save(out_path, noise_data)

def main():
    periods = 10 # Number of TRs
    overlap = 0.0

    data_paths = glob.glob("current_study_data_raw/H091/H091_scan.vhdr")
    print(data_paths)
    print('----Subject Data Segmentation----')
    for subject_vhdr in data_paths:
        print(subject_vhdr)
        eeg_read(subject_vhdr, periods, overlap)
    
    #print('----Watermelon Data Segmentation----')
    #eeg_read_noise(periods, overlap)
    

if __name__ == '__main__':
    OUTPUT_DIR = Path("data_segmented_one_subject_10_TR")
    main()
