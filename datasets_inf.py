from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

# --- EEG adjacency map (24 electrodes only) ---
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

    # parieto-occipital / occipital
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


class EEGDenoiseDatasetInf(Dataset):
    """
    Dataset for EEG/ECG denoising.
    Supports both 24-channel EEG and optional ECG (25th channel).
    """

    def __init__(
        self,
        root,
        use_adjacent: bool = True,
        seed: int = 0,
        channels: str = "single",
        main_channel: int = 0,    # 0–23 = EEG; 24 = ECG (if exists)
    ):
        
        root = Path(root)
        contaminated = np.load(root / "contaminated.npy", mmap_mode="r")   # (N_real, C, T)

        n_channels = contaminated.shape[1]
        print(f"Loaded {n_channels} channels from dataset.")

        self.contaminated = contaminated
        self.use_adjacent = use_adjacent
        self.seed = int(seed)
        self.channels = channels
        self.main_channel = int(main_channel)
        self.n_real = contaminated.shape[0]
        self.n_ch = n_channels

        # Check if channel is ECG:
        if main_channel==24:
            print('ECG Channel Detected')
            self.is_ecg = True
        else:
            print('EEG Channel Detected')
            self.is_ecg = False
        
        # --- Precompute adjacency for EEG only ---
        if self.use_adjacent:
            self.adj = []
            idx_map = {name: i for i, name in enumerate(chs)}
            for ch_name in chs:
                neighbors = closest_neighbors[ch_name]
                self.adj.append(tuple(idx_map[n] for n in neighbors))

        # --- Index mapping ---
        selected_epochs = range(self.n_real)
        if self.channels == "single":
            self.index_map = [(e, self.main_channel) for e in selected_epochs]
        else:
            self.index_map = [(e, ch) for e in selected_epochs for ch in range(self.n_ch)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        idx0, ch = self.index_map[idx]
        if self.is_ecg:
            # ECG has no spatial adjacency
            use_adj = False
        else:
            use_adj = self.use_adjacent

        cont_ep = self.contaminated[idx0, ch]
        if use_adj:
            a1, a2, a3, a4 = self.adj[ch]
            adj1 = self.contaminated[idx0, a1]
            adj2 = self.contaminated[idx0, a2]
            adj3 = self.contaminated[idx0, a3]
            adj4 = self.contaminated[idx0, a4]
            x = np.stack([cont_ep, adj1, adj2, adj3, adj4], axis=0)
        else:
            x = cont_ep[None]
        y = np.zeros_like(cont_ep[None], dtype=np.float32)
        noise_i = -1
        real_idx = idx0

        # -------------------------------
        # Normalization
        # -------------------------------
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
