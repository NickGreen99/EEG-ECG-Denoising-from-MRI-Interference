import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import EEGDenoiseDataset
from torch.utils.data import DataLoader
from pathlib import Path
from resunet import DeepDSP_UNetRes
 

val_ds = EEGDenoiseDataset(Path("/content/drive/MyDrive/data_segmented/data_segmented/23"))
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                    num_workers=2, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepDSP_UNetRes(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("best_model.pt"))

with torch.no_grad():
    for batch_idx, (x, y, clean_i, noise_i, m, sd, ch) in enumerate(val_dl):
        if batch_idx == 5:      # show first 10 batches
            break
        x, y = x.to(device), y.to(device)
        cleaned = model(x)
        print(ch)
        # ── draw *one* example from the current batch (index 0) ─────────────
        b      = 0
        t      = np.arange(y.shape[-1])

        clean_epoch  = clean_i[b].item()
        noise_epoch  = noise_i[b].item()

        x_denorm = x[b,0].cpu() * sd[b].cpu() + m[b].cpu()  # shape (3, T)
        plt.figure(figsize=(10, 4))
        plt.plot(t, y[b, 0].cpu(),           label="Ground-truth clean")
        plt.plot(t, cleaned[b, 0].cpu(),     label="Model cleaned")
        plt.plot(t, x[b,0].cpu(), alpha=.4, label="Contaminated (input)")
        plt.title(f"Clean Epoch {clean_epoch} • Noise Epoch {noise_epoch}")
        plt.xlabel("Time (samples)")
        plt.ylabel("µV")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()
