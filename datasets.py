from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


# closest_neighbors = {
#     'Fp1': ['Fp2', 'FPz'], 'Fp2': ['Fp1', 'FPz'], 'F3': ['F7', 'Fz'],
#     'F4': ['Fz', 'F8'],'C3': ['T7', 'Cz'],'C4': ['Cz', 'T8'], 'P3': ['P7', 'Pz'],
#     'P4': ['Pz', 'P8'],'O1': ['POz', 'Oz'], 'O2': ['POz', 'Oz'], 'F7': ['F3', 'FT9'],
#     'F8': ['F4', 'FT10'],'T7': ['FT9', 'C3'],'T8': ['C4', 'FT10'],'P7': ["TP9'", 'P3'],
#     'P8': ['P4', "TP10'"], 'FPz': ['Fp1', 'Fp2'], 'Fz': ['F3', 'F4'], 'Cz': ['C3', 'C4'],
#     'Pz': ['P3', 'P4'], 'POz': ['O1', 'O2'], 'Oz': ['O1', 'O2'], 'FT9': ['F7', 'T7'], 
#     'FT10': ['F8', 'T8'],"TP9'": ['T7', 'P7'], "TP10'": ['T8', 'P8']
# }
closest_neighbors = {
    # anterior row
    'Fp1' : ['Fp2', 'FPz', 'F3',  'F7'],
    'Fp2' : ['Fp1', 'FPz', 'F4',  'F8'],
    'FPz' : ['Fp1', 'Fp2', 'F3',  'F4'],

    # frontal
    'F3'  : ['Fp1', 'F7',  'Fz',  'C3'],
    'F4'  : ['Fp2', 'F8',  'Fz',  'C4'],
    'F7'  : ['Fp1', 'F3',  'FT9', 'T7'],
    'F8'  : ['Fp2', 'F4',  'FT10','T8'],
    'Fz'  : ['FPz', 'F3',  'F4',  'Cz'],

    # central
    'C3'  : ['F3',  'T7',  'Cz',  'P3'],
    'C4'  : ['F4',  'Cz',  'T8',  'P4'],
    'Cz'  : ['Fz',  'C3',  'C4',  'Pz'],

    # temporal
    'T7'  : ['FT9', 'C3',  'P7',  "TP9'"],
    'T8'  : ['FT10','C4',  'P8',  "TP10'"],
    'FT9' : ['F7',  'T7',  "TP9'", 'Fp1'],
    'FT10': ['F8',  'T8',  "TP10'",'Fp2'],
    "TP9'": ['T7',  'P7',  'FT9', 'O1'],
    "TP10'":['T8',  'P8',  'FT10','O2'],

    # parietal
    'P3'  : ['C3',  'P7',  'Pz',  'O1'],
    'P4'  : ['C4',  'P8',  'Pz',  'O2'],
    'P7'  : ['T7',  'P3',  "TP9'", 'O1'],
    'P8'  : ['T8',  'P4',  "TP10'",'O2'],
    'Pz'  : ['P3',  'P4',  'Cz',  'POz'],

    # parieto‑occipital / occipital
    'POz' : ['Pz',  'O1',  'O2',  'Oz'],
    'O1'  : ['POz', 'Oz',  'P3',  'P7'],
    'O2'  : ['POz', 'Oz',  'P4',  'P8'],
    'Oz'  : ['O1',  'O2',  'POz', 'Pz'],
}


chs = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','FPz','Fz','Cz','Pz',
    'POz','Oz','FT9','FT10',"TP9'","TP10'"
    ]
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

class EEGDenoiseDatasetOverfit(Dataset):
    def __init__(self, root, deterministic=False, use_adjacent=True):
        root = Path(root)

        noise = np.load(root.parent / "interference.npy", mmap_mode="r")
        clean = np.load(root / "clean.npy", mmap_mode="r")  # shape (N_clean, 26, T)
        contaminated = np.load(root / "contaminated.npy", mmap_mode="r")  # shape (N_contaminated, 26, T)

        clean = clean#clean[:4]
        noise = noise#noise[:4]
        contaminated = contaminated#contaminated[:4]


        #assert clean.shape[1] == noise.shape[1] == 26
        self.clean = clean
        self.noise = noise
        self.contaminated = contaminated
        
        self.n_clean, self.n_ch, self.T = clean.shape
        self.n_noise = noise.shape[0]
        self.deterministic = deterministic
        self.use_adjacent = use_adjacent

        # Precompute adjacency indices
        self.adj = []
        for ch_name in chs:
            a1, a2, a3, a4 = closest_neighbors[ch_name]
            self.adj.append( (chs.index(a1), chs.index(a2), chs.index(a3), chs.index(a4)) )

        # (epoch, channel)
        self.index_map = [(i, 0) for i in range(self.n_clean)]# for ch in range(self.n_ch)]
        self.contaminated_index_map =  [i for i in range(self.n_noise)]
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        epoch_i, ch = self.index_map[idx]
        if self.deterministic:
            noise_i = epoch_i
        else:
            #noise_i = np.random.randint(0, self.noise.shape[0])
            noise_i = self.contaminated_index_map[idx]

        clean_ep = self.clean[epoch_i, ch]
        noise_ep = self.noise[noise_i, ch]
        contaminated_ep = self.contaminated[noise_i, ch]

        if self.use_adjacent:
            a1, a2, a3, a4 = self.adj[ch]
            adj1 = self.noise[noise_i, a1]
            adj2 = self.noise[noise_i, a2]
            adj3 = self.noise[noise_i, a3]
            adj4 = self.noise[noise_i, a4]

            # For Training
            #x = np.stack([clean_ep + noise_ep, adj1, adj2, adj3, adj4], axis=0).astype(np.float32)

            # For Inference
            x = np.stack([contaminated_ep], axis=0).astype(np.float32)
        else:
            x = np.stack([clean_ep + noise_ep], axis=0).astype(np.float32)
        
        y = clean_ep[None].astype(np.float32)


        # Per-sample normalization applied to *both* x and y
        m = x.mean()
        sd = x.std() + 1e-8
        x_norm = (x - m) / sd
        y_norm = (y - m) / sd

        return (torch.from_numpy(x_norm),
                torch.from_numpy(y_norm),
                torch.tensor(epoch_i),
                torch.tensor(noise_i),
                torch.tensor(m, dtype=torch.float32),
                torch.tensor(sd, dtype=torch.float32),
                torch.tensor(ch))



# For Training
class EEGDenoiseDataset(Dataset):
    """Each clean window is randomly paired with a noisy window from the same subject."""
    def __init__(self, roots):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        # Load watermelon data
        # noise_root_dir="/content/drive/MyDrive/data_segmented/data_segmented"
        # noise_data = noise_root_dir / "watermelon.npy"
        noise = np.load("data_segmented_one_subject_3_TR/watermelon.npy", mmap_mode="r")


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
        #noise_i  = np.random.randint(0, s["n_noise"])

        # clean and noise segment from the same channel
        clean_ep = s["clean"][clean_i, ch]
        noise_ep = s["noise"][clean_i,  ch]

        # find this channel's adjacent channels
        adjacent_ch1 = chs.index(closest_neighbors[chs[ch]][0])
        adjacent_ch2 = chs.index(closest_neighbors[chs[ch]][1])

        # get noise from the adjacent channels
        adjacent_ch1_noise_ep = s["noise"][clean_i, adjacent_ch1]
        adjacent_ch2_noise_ep = s["noise"][clean_i, adjacent_ch2]

        # input matrix
        x = np.stack([clean_ep + noise_ep,
                       adjacent_ch1_noise_ep,
                       adjacent_ch2_noise_ep], axis=0).astype(np.float32)
        #x = np.stack([clean_ep + noise_ep], axis=0).astype(np.float32)

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
