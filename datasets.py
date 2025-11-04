from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

# All sensors
# closest_neighbors = {"Fp1": [
#   'F3','FPz','F7','Fz','Fp2',
#   'C3','F4','T7','Cz','FT9',
#   'C4','P3','F8','P7','Pz',
#   'P4','T8','POz','O1','Oz',
#   'P8','FT10','O2'
# ]}

# 5 sensors
# closest_neighbors = {
#  'Fp1':  ['F3','FPz','F7','Fz','Fp2'],
#  'Fp2':  ['F4','FPz','F8','Fz','Fp1'],

#  'F3':   ['Fp1','C3','F7','Fz','T7'],
#  'F4':   ['Fp2','C4','F8','Fz','T8'],

#  'C3':   ['F3','P3','T7','Cz','F7'],
#  'C4':   ['F4','P4','T8','Cz','F8'],

#  'P3':   ['C3','O1','P7','Pz','POz'],
#  'P4':   ['C4','O2','P8','Pz','POz'],

#  'O1':   ['P3','Oz','POz','P7','Pz'],
#  'O2':   ['P4','Oz','POz','P8','Pz'],

#  'F7':   ['F3','T7','FT9','Fp1','C3'],
#  'F8':   ['F4','T8','FT10','Fp2','C4'],

#  'T7':   ['C3','F7','P7','FT9','F3'],
#  'T8':   ['C4','F8','P8','FT10','F4'],

#  'P7':   ['P3','T7','C3','O1','FT9'],
#  'P8':   ['P4','T8','C4','O2','FT10'],

#  'FPz':  ['Fp1','Fp2','Fz','F3','F4'],
#  'Fz':   ['F3','F4','FPz','Cz','Fp1'],
#  'Cz':   ['C3','C4','Fz','Pz','F3'],
#  'Pz':   ['POz','P3','P4','Cz','Oz'],
#  'POz':  ['Pz','Oz','P3','P4','O1'],
#  'Oz':   ['POz','O1','O2','Pz','P3'],

#  'FT9':  ['F7','T7','P7','F3','C3'],
#  'FT10': ['F8','T8','P8','F4','C4'],
# }

# 4 sensors
closest_neighbors = {
    
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
    'T7'  : ['FT9', 'C3',  'P7',  'F7'],
    'T8'  : ['FT10','C4',  'P8',  'F8'],
    'FT9' : ['F7',  'T7',  'P7', 'Fp1'],
    'FT10': ['F8',  'T8',  'P8','Fp2'],

    # parietal
    'P3'  : ['C3',  'P7',  'Pz',  'O1'],
    'P4'  : ['C4',  'P8',  'Pz',  'O2'],
    'P7'  : ['T7',  'P3',  'C3', 'O1'],
    'P8'  : ['T8',  'P4',  'C4','O2'],
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
    'POz','Oz','FT9','FT10'
    ]

class EEGDenoiseDataset(Dataset):
    """
    Dataset for EEG denoising experiments.
    Modes:
      - "train":    Synthetic training: x = clean + noise (+ adjacents),      y = clean
      - "val_syn":  Synthetic validation: same as train but deterministic pairing
      - "val_real": Real validation:      x = contaminated (+ adjacents),     y = contaminated - interference
      - "inference":For inference only:   x = contaminated (+ adjacents),     y = dummy zeros
    """

    def __init__(
        self,
        root,
        use_adjacent: bool = True,
        split_ratio: float = 0.9,
        mode: str = "train",            # "train" | "val_syn" | "val_real" | "inference"
        seed: int = 0,
        channels: str = "single",       # "single" or "all"
        main_channel: int = 0           # which channel to learn/denoise (0 == 'Fp1' with your chs list)
    ):
        assert mode in ("train", "val_syn", "val_real", "inference")
        root = Path(root)

        # Load data arrays
        noise = np.load(root / "interference.npy",  mmap_mode="r")      # shape: (N_real, C, T)
        clean = np.load(root / "clean.npy",         mmap_mode="r")      # shape: (N_clean, C, T)
        contaminated = np.load(root / "contaminated.npy", mmap_mode="r")# shape: (N_real, C, T)

        # Sanity checks
        assert clean.shape[1] == noise.shape[1] == contaminated.shape[1] == 24, \
            f"Channel mismatch: clean={clean.shape}, noise={noise.shape}, cont={contaminated.shape}"
        assert contaminated.shape[0] == noise.shape[0], \
            "Contaminated & interference must have same number of windows."

        # Store data
        self.clean = clean
        self.noise = noise
        self.contaminated = contaminated

        self.mode = mode
        self.use_adjacent = use_adjacent
        self.seed = int(seed)

        self.n_clean, self.n_ch, self.T = clean.shape     # Clean epochs count
        self.n_real = contaminated.shape[0]               # Real (contaminated/noise) epochs count
        self.n_noise = noise.shape[0]                     # Noise epochs (same as contaminated)

        self.channels = channels
        self.main_channel = int(main_channel)
        assert 0 <= self.main_channel < self.n_ch, "main_channel out of range"
        
        # Precompute adjacency indices for all 24 channels
        if self.use_adjacent:
            self.adj = []
            idx_map = {name: i for i, name in enumerate(chs)}
            for ch_name in chs:
                neighbors = closest_neighbors[ch_name]
                print(ch_name)
                print(neighbors)
                self.adj.append(tuple(idx_map[n] for n in neighbors))
                break

        # -------- Index mapping --------
        # Train/val_syn: iterate over CLEAN epochs; val_real/inference: iterate over REAL epochs
        n_train_clean = int(split_ratio * self.n_clean)
        if mode == "train":
            selected_epochs = range(0, n_train_clean)
        elif mode == "val_syn":
            selected_epochs = range(n_train_clean, self.n_clean)
        else:  # val_real / inference iterate over all real pairs
            selected_epochs = range(self.n_real)

        if self.channels == "single":
            # only (epoch, main_channel)
            self.index_map = [(e, self.main_channel) for e in selected_epochs]
        else:
            # all channels
            self.index_map = [(e, ch) for e in selected_epochs for ch in range(self.n_ch)]

        # -------- Noise sampling state --------
        # Noise pool is (n_real × 24 channels). Because we pick noise per-channel,
        # you effectively have **67,392 noise-channel samples** for training.
        # Clean is smaller (427 × 24 = 10,248 samples), so clean windows are reused more often.
        # The permutation below ensures each **noise window** is seen each epoch before repeats.
        self._noise_perm = np.arange(self.n_noise)
        self._noise_ptr = 0
        self._epoch_k = 0

    # Call this once at the start of each training epoch
    def set_epoch(self, k: int):
        """Shuffle the noise window permutation once per training epoch."""
        self._epoch_k = int(k)
        if self.mode == "train":
            rng = np.random.default_rng(self.seed + k)
            rng.shuffle(self._noise_perm)
            self._noise_ptr = 0

    def _next_noise_index(self):
        """Fetch the next noise index from the per-epoch permutation."""
        i = int(self._noise_perm[self._noise_ptr])
        self._noise_ptr += 1
        if self._noise_ptr >= len(self._noise_perm):
            self._noise_ptr = 0
        return i

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Return a single sample: normalized input (x), normalized target (y),
        and metadata (epoch index, noise index, mean, std, channel index, real index).
        """
        idx0, ch = self.index_map[idx]

        if self.mode in ("train", "val_syn"):
            # Synthetic sample
            clean_ep = self.clean[idx0, ch]
            if self.mode == "train":
                noise_i = self._next_noise_index()
            else:
                stride = 17  # fixed stride for reproducibility
                noise_i = (idx0 * stride) % self.n_noise
            noise_ep = self.noise[noise_i, ch]

            x_main = clean_ep + noise_ep
            y_main = clean_ep

            # Adjacent channels (synthetic contamination)
            if self.use_adjacent:
                a1, a2, a3, a4  = self.adj[ch]
                adj1 = self.clean[idx0, a1] + self.noise[noise_i, a1]
                adj2 = self.clean[idx0, a2] + self.noise[noise_i, a2]
                adj3 = self.clean[idx0, a3] + self.noise[noise_i, a3]
                adj4 = self.clean[idx0, a4] + self.noise[noise_i, a4]
                x = np.stack([x_main, adj1, adj2, adj3, adj4], axis=0)
            else:
                x = x_main[None]

            y = y_main[None]
            real_idx = -1

        elif self.mode == "val_real":
            # Real contaminated → pseudo-clean
            cont_ep = self.contaminated[idx0, ch]
            intf_ep = self.noise[idx0, ch]
            y_main = cont_ep - intf_ep

            if self.use_adjacent:
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23 = self.adj[ch]
                adj1 = self.contaminated[idx0, a1]
                adj2 = self.contaminated[idx0, a2]
                adj3 = self.contaminated[idx0, a3]
                adj4 = self.contaminated[idx0, a4]
                adj5 = self.contaminated[idx0, a5]
                adj6 = self.contaminated[idx0, a6]
                adj7 = self.contaminated[idx0, a7]
                adj8 = self.contaminated[idx0, a8]
                adj9 = self.contaminated[idx0, a9]
                adj10 = self.contaminated[idx0, a10]
                adj11 = self.contaminated[idx0, a11]
                adj12 = self.contaminated[idx0, a12]
                adj13 = self.contaminated[idx0, a13]
                adj14 = self.contaminated[idx0, a14]
                adj15 = self.contaminated[idx0, a15]
                adj16 = self.contaminated[idx0, a16]
                adj17 = self.contaminated[idx0, a17]
                adj18 = self.contaminated[idx0, a18]
                adj19 = self.contaminated[idx0, a19]
                adj20 = self.contaminated[idx0, a20]
                adj21 = self.contaminated[idx0, a21]
                adj22 = self.contaminated[idx0, a22]
                adj23 = self.contaminated[idx0, a23]
                x = np.stack([cont_ep, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9, adj10,
                                adj11, adj12, adj13, adj14, adj15, adj16, adj17, adj18, adj19, adj20, adj21, adj22, adj23], axis=0)
            else:
                x = cont_ep[None]

            y = y_main[None]
            noise_i = idx0
            real_idx = idx0

        else:  # inference
            cont_ep = self.contaminated[idx0, ch]
            if self.use_adjacent:
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23 = self.adj[ch]
                adj1 = self.contaminated[idx0, a1]
                adj2 = self.contaminated[idx0, a2]
                adj3 = self.contaminated[idx0, a3]
                adj4 = self.contaminated[idx0, a4]
                adj5 = self.contaminated[idx0, a5]
                adj6 = self.contaminated[idx0, a6]
                adj7 = self.contaminated[idx0, a7]
                adj8 = self.contaminated[idx0, a8]
                adj9 = self.contaminated[idx0, a9]
                adj10 = self.contaminated[idx0, a10]
                adj11 = self.contaminated[idx0, a11]
                adj12 = self.contaminated[idx0, a12]
                adj13 = self.contaminated[idx0, a13]
                adj14 = self.contaminated[idx0, a14]
                adj15 = self.contaminated[idx0, a15]
                adj16 = self.contaminated[idx0, a16]
                adj17 = self.contaminated[idx0, a17]
                adj18 = self.contaminated[idx0, a18]
                adj19 = self.contaminated[idx0, a19]
                adj20 = self.contaminated[idx0, a20]
                adj21 = self.contaminated[idx0, a21]
                adj22 = self.contaminated[idx0, a22]
                adj23 = self.contaminated[idx0, a23]
                x = np.stack([cont_ep, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9, adj10,
                                adj11, adj12, adj13, adj14, adj15, adj16, adj17, adj18, adj19, adj20, adj21, adj22, adj23], axis=0)
            else:
                x = cont_ep[None]
            y = np.zeros_like(cont_ep[None], dtype=np.float32)
            noise_i = -1
            real_idx = idx0

        # Normalize
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        m = x.mean()
        sd = x.std() + 1e-8
        x_norm = (x - m) / sd
        y_norm = (y - m) / sd

        return (
            torch.from_numpy(x_norm),
            torch.from_numpy(y_norm),
            torch.tensor(idx0),
            torch.tensor(noise_i),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32),
            torch.tensor(ch),
            torch.tensor(real_idx),
        )

# class EEGDenoiseDataset(Dataset):
#     def __init__(self, root, use_adjacent=True, split_ratio=0.9, split="train"):
#         root = Path(root)

#         noise = np.load(root / "interference.npy", mmap_mode="r")
#         clean = np.load(root / "clean.npy", mmap_mode="r")  # shape (N_clean, 24, T)
#         contaminated = np.load(root / "contaminated.npy", mmap_mode="r")  # shape (N_contaminated, 24, T)

#         assert clean.shape[1] == noise.shape[1] == contaminated.shape[1] == 24
        
#         self.clean = clean
#         self.noise = noise
#         self.contaminated = contaminated
        
#         self.split = split

#         self.n_clean, self.n_ch, self.T = clean.shape
#         self.use_adjacent = use_adjacent

#         # Precompute adjacency indices
#         if use_adjacent == True:
#             self.adj = []
#             for ch_name in chs:
#                 a1, a2, a3, a4 = closest_neighbors[ch_name]
#                 self.adj.append( (chs.index(a1), chs.index(a2), chs.index(a3), chs.index(a4)) )

#         # Split train/test by epochs (but keep all channels for each epoch)
#         n_train = int(split_ratio * self.n_clean)
#         if split == "train":
#             print(f'Train Split is from: 0 to {n_train}')
#             selected_epochs = range(0, n_train)
#         else:
#             print(f'Val/Inference Split is from: {n_train} to {self.n_clean}')
#             selected_epochs = range(n_train, self.n_clean)
        
#         # (epoch, channel)
#         self.index_map = [(e, 0) for e in selected_epochs ]#for ch in range(self.n_ch)]
    
#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, idx):
#         epoch_i, ch = self.index_map[idx]
#         noise_i = np.random.randint(0, self.noise.shape[0])

#         clean_ep = self.clean[epoch_i, ch]
#         noise_ep = self.noise[noise_i, ch]
#         contaminated_ep = self.contaminated[noise_i, ch]

#         val_noise_i = epoch_i if epoch_i < len(self.contaminated) else epoch_i % len(self.contaminated)
#         val_contaminated_ep = self.contaminated[val_noise_i, ch]
#         val_noise_ep = self.noise[val_noise_i, ch]
        
#         if self.use_adjacent:
#             print('Adjacent Sensors mode...\n')
#             a1, a2, a3, a4 = self.adj[ch]
#             adj1 = self.noise[noise_i, a1]
#             adj2 = self.noise[noise_i, a2]
#             adj3 = self.noise[noise_i, a3]
#             adj4 = self.noise[noise_i, a4]

#             if self.split == "train":
#                 # For Training
#                 print('Training mode...\n')
#                 x = np.stack([clean_ep + noise_ep, adj1, adj2, adj3, adj4], axis=0).astype(np.float32)
#                 y = clean_ep[None].astype(np.float32)

#             elif self.split == "val":
#                 # For validation
#                 print('Validation mode...\n')
#                 a1, a2, a3, a4 = self.adj[ch]
#                 adj1 = self.contaminated[val_noise_i, a1]
#                 adj2 = self.contaminated[val_noise_i, a2]
#                 adj3 = self.contaminated[val_noise_i, a3]
#                 adj4 = self.contaminated[val_noise_i, a4]

#                 x = np.stack([val_contaminated_ep, adj1, adj2, adj3, adj4], axis=0).astype(np.float32)
#                 y = val_contaminated_ep[None].astype(np.float32) - val_noise_ep[None].astype(np.float32)
                
#             else:
#                 # For Inference
#                 print('Inference mode...\n')
#                 x = np.stack([contaminated_ep, adj1, adj2, adj3, adj4], axis=0).astype(np.float32)
#                 y = clean_ep[None].astype(np.float32)

#         else:
#             print('One Sensor mode...\n')
#             if self.split == "train":
#                 print('Training mode...\n')
#                 x = np.stack([clean_ep + noise_ep], axis=0).astype(np.float32)
#                 y = clean_ep[None].astype(np.float32)

#             elif self.split == "val":
#                 print('Validation mode...\n')
#                 # For validation
#                 x = np.stack([val_contaminated_ep], axis=0).astype(np.float32)
#                 y = val_contaminated_ep[None].astype(np.float32) - val_noise_ep[None].astype(np.float32)

#             else:
#                 # For Inference
#                 print('Inference mode...\n')
#                 x = np.stack([contaminated_ep], axis=0).astype(np.float32)
#                 y = clean_ep[None].astype(np.float32)

#         # Per-sample normalization applied to *both* x and y
#         m = x.mean()
#         sd = x.std() + 1e-8
#         x_norm = (x - m) / sd
#         y_norm = (y - m) / sd

#         return (torch.from_numpy(x_norm),
#                 torch.from_numpy(y_norm),
#                 torch.tensor(epoch_i),
#                 torch.tensor(noise_i),
#                 torch.tensor(m, dtype=torch.float32),
#                 torch.tensor(sd, dtype=torch.float32),
#                 torch.tensor(ch))
