import mne
import numpy as np
from subject_segments_all_cleaned import subject_segments
import glob
import re
import gc
from pathlib import Path
from mne.preprocessing import ICA


def get_logpsd(inst, picks, **psd_args):
    spec = inst.compute_psd(picks=picks, **psd_args)
    psd, freqs = spec.get_data(return_freqs=True)
    return freqs, np.log(psd + 1e-12)  # (n_traces, n_freq)

def zscore_freq(logpsd):
    m  = logpsd.mean(axis=1, keepdims=True)
    sd = logpsd.std(axis=1, keepdims=True) + 1e-12
    return (logpsd - m) / sd


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
    noise_segments = []

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

        contaminated_seg_ica = ICA(n_components=None, max_iter="auto", random_state=97)
        contaminated_seg_ica.fit(contaminated_seg)
        
        # 1) Watermelon template (robust): average across sensors, then z-score
        fw, wm_logpsd_all = get_logpsd(watermelon, picks='eeg', method='welch', fmin=20, fmax=250, n_fft=4096)
        wm_template = zscore_freq(wm_logpsd_all).mean(axis=0)  # 1D (n_freq,)

        # 2) Human IC spectra (sources from ICA)
        sources = contaminated_seg_ica.get_sources(raw)
        fr, ic_logpsd = get_logpsd(sources, picks='all', method='welch', fmin=20, fmax=250, n_fft=4096)
        assert np.allclose(fr, fw)

        ic_z = zscore_freq(ic_logpsd)

        # 3) Rank ICs by correlation (shape); optionally combine with LSD if you want
        def corr1d(a,b):
            a0, b0 = a - a.mean(), b - b.mean()
            return float(np.dot(a0,b0)/(np.linalg.norm(a0)*np.linalg.norm(b0)+1e-12))

        scores = [corr1d(wm_template, ic_z[j]) for j in range(ic_z.shape[0])]
        topk_idx = np.argsort(scores)[-14:][::-1]
        
        # Define the full range
        full_range = set(range(24))  # 0 to 23

        # Get the complement
        complementary_list = sorted(full_range - set(topk_idx))

        contaminated_seg_ica.exclude = complementary_list
        noise_proxy = contaminated_seg.copy()
        contaminated_seg_ica.apply(noise_proxy)

        noise_proxy_data = epoch_data(noise_proxy, window_size_time, overlap)
        noise_segments.append(noise_proxy_data)

        del contaminated_seg, contaminated_data, noise_proxy_data, noise_proxy 
        gc.collect()

    # Concatenate all epochs along axis 0 (n_epochs)
    contaminated_segments = np.concatenate(contaminated_segments, axis=0)
    noise_segments = np.concatenate(noise_segments, axis=0)

    # Save subject's contaminated.npy
    out_path = SUBJECT_DIR / 'contaminated.npy'
    np.save(out_path, contaminated_segments)
    out_path = SUBJECT_DIR / 'interference.npy'
    np.save(out_path, noise_segments)
       
    del contaminated_segments, noise_segments
    gc.collect()

def main():
    periods = 10 # Number of TRs
    overlap = 0.0

    data_paths = glob.glob("current_study_data_raw/H091/H091_scan.vhdr")
    print(data_paths)
    print('----Subject Data Segmentation----')
    for subject_vhdr in data_paths:
        eeg_read(subject_vhdr, periods, overlap)

if __name__ == '__main__':

    watermelon = mne.io.read_raw_brainvision('current_study_data_raw/watermelon2.0/watermelon2.0_scan.vhdr', preload=True)

    chs = watermelon.info["ch_names"][0:24]

    watermelon.pick(chs)

    watermelon.crop(tmin=1855.415, tmax=2813.099)
    watermelon.set_annotations(None)
    watermelon.resample(512.)
    watermelon = watermelon.copy().filter(l_freq=0.5, h_freq=None, verbose=False)

    OUTPUT_DIR = Path("data_segmented_one_subject_10_TR_all_channels")
    main()
