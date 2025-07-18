from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


closest_neighbors = {
    'Fp1': ['Fp2', 'FPz'], 'Fp2': ['Fp1', 'FPz'], 'F3': ['F7', 'Fz'],
    'F4': ['Fz', 'F8'],'C3': ['T7', 'Cz'],'C4': ['Cz', 'T8'], 'P3': ['P7', 'Pz'],
    'P4': ['Pz', 'P8'],'O1': ['POz', 'Oz'], 'O2': ['POz', 'Oz'], 'F7': ['F3', 'FT9'],
    'F8': ['F4', 'FT10'],'T7': ['FT9', 'C3'],'T8': ['C4', 'FT10'],'P7': ["TP9'", 'P3'],
    'P8': ['P4', "TP10'"], 'FPz': ['Fp1', 'Fp2'], 'Fz': ['F3', 'F4'], 'Cz': ['C3', 'C4'],
    'Pz': ['P3', 'P4'], 'POz': ['O1', 'O2'], 'Oz': ['O1', 'O2'], 'FT9': ['F7', 'T7'], 
    'FT10': ['F8', 'T8'],"TP9'": ['T7', 'P7'], "TP10'": ['T8', 'P8']
}

chs = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','FPz','Fz','Cz','Pz',
    'POz','Oz','FT9','FT10',"TP9'","TP10'"
    ]

# For Training
class EEGDenoiseDataset(Dataset):
    """Each clean window is randomly paired with a noisy window from the same subject."""
    def __init__(self, roots):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        # Load watermelon data
        # noise_root_dir="/content/drive/MyDrive/data_segmented/data_segmented"
        # noise_data = noise_root_dir / "watermelon.npy"
        noise = np.load("data_segmented_3/watermelon.npy", mmap_mode="r")


        self.samples = []
        for subj in roots:
            subj = Path(subj)
            try:
                clean = np.load(subj / "clean.npy", mmap_mode="r")
            except FileNotFoundError as e:
                print(f"Skipping {subj.name}: {e}")
                continue
            self.samples.append({
                "clean":   clean,
                "noise":   noise,
                "n_clean": clean.shape[0],
                "n_noise": noise.shape[0],
                "n_ch":    26
                #"name":    subj.name
            })

        # subject → clean-epoch → channel
        self.index_map = [
            (subj_idx, i, ch)
            for subj_idx, s in enumerate(self.samples)
            for i        in range(s["n_clean"])
            for ch       in range(s["n_ch"])
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subj_idx, clean_i, ch = self.index_map[idx]
        s = self.samples[subj_idx]

        # pick a random noise epoch *and remember its index*
        noise_i  = np.random.randint(0, s["n_noise"])

        # clean and noise segment from the same channel
        clean_ep = s["clean"][clean_i, ch]
        noise_ep = s["noise"][noise_i,  ch]

        # find this channel's adjacent channels
        adjacent_ch1 = chs.index(closest_neighbors[chs[ch]][0])
        adjacent_ch2 = chs.index(closest_neighbors[chs[ch]][1])

        # get noise from the adjacent channels
        adjacent_ch1_noise_ep = s["noise"][noise_i, adjacent_ch1]
        adjacent_ch2_noise_ep = s["noise"][noise_i, adjacent_ch2]

        # input matrix
        # x = np.stack([clean_ep + noise_ep,
        #               adjacent_ch1_noise_ep,
        #               adjacent_ch2_noise_ep], axis=0).astype(np.float32)
        x = np.stack([clean_ep + noise_ep], axis=0).astype(np.float32)
        # target
        y = clean_ep[None].astype(np.float32)

        # normalize input
        m, sd = x.mean(), x.std() + 1e-8
        x_norm = (x - m) / sd

        return (
            torch.from_numpy(x_norm),          # mixed+noise input              (shape 2×T)
            torch.from_numpy(y),          # clean target                   (shape 1×T)
            torch.tensor(clean_i),
            torch.tensor(noise_i),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32),
            torch.tensor(ch)
        )

# For Inference
class EEGDenoiseDatasetContaminated(Dataset):
    """Each clean window is randomly paired with a noisy window from the same subject."""
    def __init__(self, roots):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        # Load watermelon data
        # noise_root_dir="/content/drive/MyDrive/data_segmented/data_segmented"
        # noise_data = noise_root_dir / "watermelon.npy"
        noise = np.load("data_segmented_3/watermelon.npy", mmap_mode="r")


        self.samples = []
        for subj in roots:
            subj = Path(subj)
            try:
                contaminated = np.load(subj / "contaminated.npy", mmap_mode="r")
            except FileNotFoundError as e:
                print(f"Skipping {subj.name}: {e}")
                continue
            self.samples.append({
                "noise":   noise,
                "contaminated": contaminated,
                "n_noise": noise.shape[0],
                "n_contaminated": contaminated.shape[0],
                "n_ch":    1
            })

        # subject → clean-epoch → channel
        self.index_map = [
            (subj_idx, i, ch)
            for subj_idx, s in enumerate(self.samples)
            for i        in range(s["n_noise"])
            for ch       in range(s["n_ch"])
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subj_idx, noise_i, ch = self.index_map[idx]
        s = self.samples[subj_idx]

        # clean and noise segment from the same channel
        noise_ep = s["noise"][noise_i,  ch]
        contam_ep = s["contaminated"][noise_i, ch]

        # find this channel's adjacent channels
        adjacent_ch1 = chs.index(closest_neighbors[chs[ch]][0])
        adjacent_ch2 = chs.index(closest_neighbors[chs[ch]][1])

        # get noise from the adjacent channels
        adjacent_ch1_noise_ep = s["noise"][noise_i, adjacent_ch1]
        adjacent_ch2_noise_ep = s["noise"][noise_i, adjacent_ch2]

        # input matrix
        x = np.stack([contam_ep,
                      adjacent_ch1_noise_ep,
                      adjacent_ch2_noise_ep], axis=0).astype(np.float32)

        # normalize input
        m, sd = x.mean(), x.std() + 1e-8
        x = (x - m) / sd

        return (
            torch.from_numpy(x),          # mixed+noise input              (shape 2×T)
            torch.tensor(noise_i),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32),
            torch.tensor(ch)
        )
