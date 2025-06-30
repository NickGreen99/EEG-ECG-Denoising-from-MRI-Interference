import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class L1L2Loss(nn.Module):
    def __init__(self, l2_weight: float = 0.1):
        super().__init__()
        self.l2_w = l2_weight
        self.l1   = nn.L1Loss()
        self.l2   = nn.MSELoss()

    def forward(self, pred, target):
        return self.l1(pred, target) + self.l2_w * self.l2(pred, target)

class EEGDenoiseDataset(Dataset):
    """Each element of `roots` is one subject folder containing clean.npy, noise.npy, dirty.npy."""
    def __init__(self, roots):
        # allow passing a single path or a list of paths
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.samples = []
        
        for subj in roots:
            subj = Path(subj)
            try:
                clean = np.load(subj / "clean.npy")   # shape (N_epochs, n_ch, T)
                noise = np.load(subj / "noise.npy")
                dirty = np.load(subj / "dirty.npy")
            except FileNotFoundError as e:
                print(f"Skipping {subj.name}: {e}")
                continue

            # for each epoch and each channel, build (noisy+clean, noise) → clean
            for d_ep, n_ep, c_ep in zip(dirty, noise, clean):
                for ch in range(d_ep.shape[0]):
                    x = np.stack([n_ep[ch]+c_ep[ch],        # use the *actual* dirty signal
                                  n_ep[ch]       # pure noise
                                 ], axis=0).astype(np.float32)
                    y = c_ep[ch][None].astype(np.float32)
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        m  = x.mean()
        sd = x.std() + 1e-8
        x = (x - m) / sd
        y = (y - m) / sd
        return torch.from_numpy(x), torch.from_numpy(y)

def train(
    root_dir="/content/drive/MyDrive/data_segmented/data_segmented",
    batch_size=32,
    epochs=10,
    lr=2e-5,
    log_dir="runs/denoise",
    val_split=0.1,
    model_save_path="best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)

    root_dir = Path(root_dir)
    subj_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    random.seed(0)
    random.shuffle(subj_dirs)

    n_val = max(1, int(val_split * len(subj_dirs)))
    val_subj   = subj_dirs[:n_val]
    train_subj = subj_dirs[n_val:]
    print(f"Training on {len(train_subj)} subjects; validating on {len(val_subj)} subjects")

    train_ds = EEGDenoiseDataset(train_subj)
    val_ds   = EEGDenoiseDataset(val_subj)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = DeepDSP_UNetRes(in_channels=2, out_channels=1).to(device)
    criterion = L1L2Loss(l2_weight=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ts = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(Path(log_dir) / ts)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred_noise = model(x)
            cleaned_output = x[:, 0:1, :] - pred_noise
            loss = criterion(cleaned_output, y)

            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/batch", loss.item(), global_step)
            running_loss += loss.item()
            global_step += 1

            if batch_idx % 500 == 0:
                print(
                    f"Epoch {epoch:02d}/{epochs}  "
                    f"Batch {batch_idx:04d}/{len(train_dl):04d}  "
                    f"Loss {loss.item():.6f}"
                )

        train_epoch_loss = running_loss / len(train_dl)
        writer.add_scalar("Loss/epoch", train_epoch_loss, epoch)
        print(f"[{epoch:02d}/{epochs}]  Train avg_loss={train_epoch_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred_noise = model(x)
                cleaned_output = x[:, 0:1, :] - pred_noise
                val_loss += criterion(cleaned_output, y).item()

        val_epoch_loss = val_loss / len(val_dl)
        writer.add_scalar("Loss/val_epoch", val_epoch_loss, epoch)
        print(f"[{epoch:02d}/{epochs}]  Validation avg_loss={val_epoch_loss:.6f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f">>> Saved new best model with val_loss={best_val_loss:.6f}")

    writer.close()

    # Visual check
    print("\nVisualizing predictions from best model...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred_noise = model(x)
            cleaned = x[:, 0:1, :] - pred_noise

            t = np.arange(y.shape[-1])
            plt.figure(figsize=(10, 4))
            plt.plot(t, y[0, 0].cpu(), label="Ground-truth clean")
            plt.plot(t, cleaned[0, 0].cpu(), label="Model cleaned")
            plt.plot(t, x[0, 0].cpu(), label="Dirty (input)", alpha=0.4)
            plt.title("EEG Denoising (1 Sample)")
            plt.xlabel("Time (samples)")
            plt.ylabel("µV")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            break

if __name__ == "__main__":
    train()
