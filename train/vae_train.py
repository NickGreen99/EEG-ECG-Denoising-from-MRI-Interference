# # vae_train.py (Adapted for TensorBoard)
# import math
# import torch
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from pathlib import Path
# import torch.nn as nn
# import time # NEW: For generating a unique run name
# from torch.utils.tensorboard import SummaryWriter # NEW: For logging

# from vae_dataset import NoiseChannelDataset
# from vae_model import NoiseVAE 

# def kl_gaussian(mu, logvar):
#     return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# # def mean_dc(x): # x: [B,1,T]
# #     return x.mean(dim=[1,2]).pow(2).mean()

# def next_multiple(x, m): # ceil to multiple of m
#     return int(math.ceil(x / m) * m)

# def feature_loss(encoder, x1, x2, layer_index=11):
#     """
#     Calculates L2 distance between intermediate feature maps of two inputs.
#     """
#     feature_extractor = nn.Sequential(*list(encoder.enc.children())[:layer_index+1])
    
#     f1 = feature_extractor(x1)
#     f2 = feature_extractor(x2)
    
#     return F.mse_loss(f1, f2)
    
# def train_noise_vae(
#     subj_dir="data_segmented_91_10_TR_all_channels/91",
#     main_channel=0,
#     z_dim=256,
#     batch_size=32,
#     epochs=1500,
#     lr=3e-4,
#     beta=1e-3, # KL weight
#     #lambda_dc=1e-4, # Mean DC loss weight
#     lambda_feat=1e-2, # Feature loss weight
#     alpha_recon=0.5, # Hybrid Recon alpha
#     weight_decay=1e-5,
#     save_path="noise_vae_rec_reg_feat.pt",
#     log_dir="runs/vae_noise_model", # NEW: Tensorboard logging directory
#     device=None
# ):
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --- TensorBoard Setup ---
#     ts = time.strftime("%Y%m%d-%H%M%S")
#     run_name = f"vae_rec_reg_feature_loss" # Create a meaningful run name
#     writer = SummaryWriter(Path(log_dir) / ts / run_name)
#     # -------------------------

#     ds = NoiseChannelDataset(subj_dir, main_channel=main_channel, mmap=True)
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

#     T_orig = ds.T
#     T_model = next_multiple(T_orig, 64)
#     vae = NoiseVAE(T=T_model, z_dim=z_dim).to(device)
#     opt = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

#     print(f"T_orig={T_orig}, T_model={T_model}, z_dim={z_dim}, New Loss: feat={lambda_feat}, alpha={alpha_recon}")
    
#     global_step = 0 # NEW: Counter for per-batch logging

#     for epoch in range(1, epochs+1):
#         vae.train()
#         # NEW: Track epoch-averaged metrics
#         running_loss_total, running_loss_recon, running_loss_kl, running_loss_feat, running_loss_tv, running_loss_dc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
#         for batch_idx, (x, m, sd) in enumerate(dl):
#             x = x.to(device) 

#             # Right-pad to T_model
#             if x.shape[-1] != T_model:
#                 pad = (0, T_model - x.shape[-1])
#                 x_pad = F.pad(x, pad, mode="reflect")
#             else:
#                 x_pad = x

#             xhat, mu, logvar = vae(x_pad)

#             # Crop decoder output back to original T for loss
#             xhat_crop = xhat[..., :T_orig]

#             # --- LOSS CALCULATION ---
#             l1_recon = F.l1_loss(xhat_crop, x)
#             l2_recon = F.mse_loss(xhat_crop, x)
#             recon_loss = alpha_recon * l1_recon + (1 - alpha_recon) * l2_recon
            
#             kl_loss = kl_gaussian(mu, logvar)
#             feat_loss_raw = feature_loss(vae, x_pad, xhat)
#             #dc_loss = mean_dc(xhat_crop)

#             # --- WEIGHTED LOSSES ---
#             loss_weighted_kl = beta * kl_loss
#             loss_weighted_feat = lambda_feat * feat_loss_raw
#             #loss_weighted_dc = lambda_dc * dc_loss
            
#             # --- FINAL LOSS COMBINATION ---
#             loss = (
#                 recon_loss 
#                 + loss_weighted_kl
#                 + loss_weighted_feat
#                 #+ loss_weighted_dc
#             )

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             # --- PER-BATCH LOGGING (NEW) ---
#             writer.add_scalar("Loss/Total/Batch", loss.item(), global_step)
#             writer.add_scalar("Loss/Recon/Batch", recon_loss.item(), global_step)
#             writer.add_scalar("Loss/KL_Weighted/Batch", loss_weighted_kl.item(), global_step)
#             writer.add_scalar("Loss/Feature_Weighted/Batch", loss_weighted_feat.item(), global_step)
#             #writer.add_scalar("Loss/DC_Weighted/Batch", loss_weighted_dc.item(), global_step)
#             global_step += 1
#             # -------------------------------

#             # Update running totals
#             running_loss_total += loss.item()
#             running_loss_recon += recon_loss.item()
#             running_loss_kl += kl_loss.item()
#             running_loss_feat += feat_loss_raw.item()
#             #running_loss_dc += dc_loss.item()

#         # --- EPOCH-AVERAGED LOGGING (NEW) ---
#         dl_len = len(dl)
#         writer.add_scalar("Loss_Epoch/Total", running_loss_total / dl_len, epoch)
#         writer.add_scalar("Loss_Epoch/Recon", running_loss_recon / dl_len, epoch)
#         writer.add_scalar("Loss_Epoch/KL_Raw", running_loss_kl / dl_len, epoch) # Log raw KL for monitoring
#         writer.add_scalar("Loss_Epoch/Feature_Raw", running_loss_feat / dl_len, epoch)
#         #writer.add_scalar("Loss_Epoch/DC_Raw", running_loss_dc / dl_len, epoch)
#         # ------------------------------------

#         if epoch % 10 == 0 or epoch == 1:
#             avg_loss = running_loss_total / dl_len
#             avg_recon = running_loss_recon / dl_len
#             avg_kl = running_loss_kl / dl_len
#             avg_feat = running_loss_feat / dl_len
            
#             print(f"[{epoch:04d}/{epochs}] loss={avg_loss:.6f}  "
#                   f"recon={avg_recon:.5f} kl={avg_kl:.5f} feat={avg_feat:.5f}")

#     torch.save({"state_dict": vae.state_dict(), "T": T_model, "z_dim": z_dim}, save_path)
#     print("Saved:", save_path)
#     writer.close() # NEW: Close writer when done

# if __name__ == "__main__":
#     train_noise_vae()


# vae_train.py
# Train a NoiseVAE on interference (noise) windows with optional clean-suppression.
# - Uses per-window z-score provided by the dataset (NO double-normalization).
# - Reflect-pads to T_model (multiple of 64), crops losses back to original T.
# - Loss = 0.7*L1 + 0.3*MSE + beta*KL + λ_dc*DC + λ_feat*feat_loss + λ_band*band_pen
# - Clean-suppression: add λ_clean * ||xhat_clean||^2 on clean windows so VAE→0 on EEG.
# - Saves checkpoint dict: {"T", "z_dim", "state_dict", "epoch", "cfg"}.

import os
import math
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vae_dataset import NoiseChannelDataset  # interference dataset: normalized per-window
from vae_model import NoiseVAE


# --------------------------- Utilities ---------------------------

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ceil_to_multiple(x, m):
    return int(math.ceil(x / m) * m)

def kl_gaussian(mu, logvar):
    # KL(q(z|x)||p(z)) for N(0, I) prior
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def build_feature_extractor_frozen(vae: NoiseVAE, upto: int = 2, device="cpu"):
    """
    Make a FROZEN copy of the first `upto+1` encoder blocks (early textures).
    We deepcopy to avoid freezing the main VAE encoder.
    """
    enc_children = list(vae.enc.children())
    upto = max(0, min(upto, len(enc_children) - 1))
    feat_net = nn.Sequential(*copy.deepcopy(enc_children[:upto + 1])).to(device).eval()
    for p in feat_net.parameters():
        p.requires_grad_(False)
    return feat_net

def feature_loss(feat_net: nn.Module, x1: torch.Tensor, x2: torch.Tensor):
    # Both tensors should be [B,1,T_model] (use padded versions for stable shapes)
    return F.mse_loss(feat_net(x1), feat_net(x2))

def band_energy_penalty(xhat: torch.Tensor, sfreq: float,
                        penalize=(1.0, 20.0)):
    """
    Penalize predicted-noise energy in the EEG band to bias toward artifact bands.
    xhat: [B,1,T_crop] (z-scored space)
    penalize: (low, high) Hz. Energy inside this band is penalized.
    """
    B, C, T = xhat.shape
    X = torch.fft.rfft(xhat, dim=-1)                           # [B,1,F]
    freqs = torch.fft.rfftfreq(T, d=1.0/sfreq).to(xhat.device) # [F]
    mask = ((freqs >= penalize[0]) & (freqs <= penalize[1])).float()
    mask = mask.view(1, 1, -1)
    power = (X.abs() ** 2)
    return (power * mask).mean()

class CleanChannelDataset(Dataset):
    """
    Loads clean.npy and returns per-window, per-sample normalized [1,T] tensors for a single channel.
    Matches the normalization used in NoiseChannelDataset.
    """
    def __init__(self, subj_dir, main_channel=0, mmap=True):
        subj_dir = Path(subj_dir)
        mode = "r" if mmap else None
        arr = np.load(subj_dir / "clean.npy", mmap_mode=mode)  # (N, C, T)
        self.arr = arr
        self.C = arr.shape[1]
        self.T = arr.shape[2]
        self.main_ch = int(main_channel)

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx, self.main_ch]           # [T] numpy
        x = x.astype(np.float32)
        m = x.mean()
        sd = x.std() + 1e-8
        x = (x - m) / sd
        x = torch.from_numpy(x[None, :])          # [1,T]
        return x, m, sd


# --------------------------- Training ---------------------------

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    set_seed(cfg.seed)

    # --- Datasets ---
    noise_ds = NoiseChannelDataset(cfg.noise_subj, main_channel=cfg.channel, mmap=True)
    T_orig = noise_ds.T
    T_model = ceil_to_multiple(T_orig, 64)
    if cfg.T_model is not None:
        T_model = int(cfg.T_model)
    print(f"[info] T_orig={T_orig}, T_model={T_model}")

    # Clean dataset (optional but recommended)
    use_clean = False
    if cfg.clean_subj is not None:
        clean_path = Path(cfg.clean_subj) / "clean.npy"
        if clean_path.exists():
            clean_ds = CleanChannelDataset(cfg.clean_subj, main_channel=cfg.channel, mmap=True)
            use_clean = True
        else:
            print(f"[warn] {clean_path} not found; training without clean-suppression.")
    else:
        print("[warn] No clean_subj provided; training without clean-suppression.")

    # --- Loaders ---
    noise_loader = DataLoader(
        noise_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=True
    )
    clean_loader = None
    clean_iter = None
    if use_clean:
        clean_loader = DataLoader(
            clean_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=True
        )
        clean_iter = iter(clean_loader)

    # --- Model ---
    vae = NoiseVAE(T=T_model, z_dim=cfg.z_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.lr)

    # --- Frozen early feature extractor (created once; refreshed later if you wish) ---
    feat_net = build_feature_extractor_frozen(vae, upto=cfg.feature_upto, device=device)

    # --- Training ---
    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        vae.train()
        # KL warmup to a small beta
        beta = cfg.beta_final * min(1.0, epoch / max(1, cfg.warmup_epochs))
        lam_feat = (cfg.lambda_feat if epoch > cfg.warmup_epochs else 0.0)

        running = {"recon": 0.0, "dc": 0.0, "feat": 0.0, "band": 0.0, "kl": 0.0, "clean": 0.0, "total": 0.0}
        n_steps = 0

        for xb_noise, _, _ in noise_loader:
            xb_noise = xb_noise.to(device)   # [B,1,T_orig] already normalized
            B, _, Tn = xb_noise.shape

            # reflect-pad to T_model if needed
            if Tn != T_model:
                x_pad = F.pad(xb_noise, (0, T_model - Tn), mode="reflect")
            else:
                x_pad = xb_noise

            # Posterior mean decode (stable training target)
            mu, logvar = vae.encode(x_pad)
            xhat = vae.decode(mu)                        # [B,1,T_model]
            xhat_c = xhat[..., :Tn]                      # crop back to original T

            # Core recon losses on cropped region (compare to xb_noise, already normalized)
            l1 = F.l1_loss(xhat_c, xb_noise)
            l2 = F.mse_loss(xhat_c, xb_noise)
            recon = 0.7 * l1 + 0.3 * l2

            # Small DC penalty on (full) prediction
            dc = cfg.lambda_dc * xhat.mean(dim=[1, 2]).pow(2).mean()

            # Early-feature loss (on padded T_model for consistent shapes), after warmup
            lf = torch.tensor(0.0, device=device)
            if lam_feat > 0.0:
                with torch.no_grad():
                    x_pad_detached = x_pad.detach()
                lf = lam_feat * feature_loss(feat_net, x_pad_detached, xhat)

            # Optional band penalty on cropped prediction to suppress EEG-band energy
            lb = torch.tensor(0.0, device=device)
            if cfg.lambda_band > 0.0:
                lb = cfg.lambda_band * band_energy_penalty(xhat_c, cfg.sfreq, penalize=tuple(cfg.band_penalize))

            # Tiny KL (warmup)
            kl = beta * kl_gaussian(mu, logvar)

            loss = recon + dc + lf + lb + kl

            # Clean-suppression term (optional)
            lc = torch.tensor(0.0, device=device)
            if use_clean and cfg.lambda_clean > 0.0:
                try:
                    xb_clean, _, _ = next(clean_iter)
                except StopIteration:
                    clean_iter = iter(clean_loader)
                    xb_clean, _, _ = next(clean_iter)
                xb_clean = xb_clean.to(device)  # normalized
                Tc = xb_clean.shape[-1]
                if Tc != T_model:
                    x_clean_pad = F.pad(xb_clean, (0, T_model - Tc), mode="reflect")
                else:
                    x_clean_pad = xb_clean

                mu_c, _ = vae.encode(x_clean_pad)
                xhat_clean = vae.decode(mu_c)[..., :Tc]   # predicted noise on clean → should be ~0
                lc = cfg.lambda_clean * (xhat_clean.pow(2).mean())
                loss = loss + lc

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(vae.parameters(), cfg.grad_clip)
            opt.step()

            # Accumulate logs
            running["recon"] += float(recon.item())
            running["dc"]    += float(dc.item())
            running["feat"]  += float(lf.item())
            running["band"]  += float(lb.item())
            running["kl"]    += float(kl.item())
            running["clean"] += float(lc.item())
            running["total"] += float(loss.item())
            n_steps += 1

        # Epoch summary
        for k in running:
            running[k] /= max(1, n_steps)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={running['total']:.5f} | recon={running['recon']:.5f} | kl={running['kl']:.5f} | "
            f"dc={running['dc']:.5f} | feat={running['feat']:.5f} | band={running['band']:.5f} | clean={running['clean']:.5f} | "
            f"beta={beta:.4g} lam_feat={lam_feat:.4g}"
        )

        # Quick sanity every few epochs (noise MSE on small batch; clean energy ratio)
        if epoch % cfg.eval_every == 0:
            with torch.no_grad():
                vae.eval()
                # Noise recon (on 1 mini-batch)
                xb_noise, _, _ = next(iter(noise_loader))
                xb_noise = xb_noise.to(device)
                Tn = xb_noise.shape[-1]
                if Tn != T_model:
                    x_pad = F.pad(xb_noise, (0, T_model - Tn), mode="reflect")
                else:
                    x_pad = xb_noise
                mu, _ = vae.encode(x_pad)
                xhat = vae.decode(mu)[..., :Tn]
                mse = F.mse_loss(xhat, xb_noise).item()
                l1 = F.l1_loss(xhat, xb_noise).item()

                msg = f"  [eval] noise: mse={mse:.5f} l1={l1:.5f}"

                if use_clean:
                    xb_clean, _, _ = next(iter(clean_loader))
                    xb_clean = xb_clean.to(device)
                    Tc = xb_clean.shape[-1]
                    if Tc != T_model:
                        x_clean_pad = F.pad(xb_clean, (0, T_model - Tc), mode="reflect")
                    else:
                        x_clean_pad = xb_clean
                    mu_c, _ = vae.encode(x_clean_pad)
                    xhat_clean = vae.decode(mu_c)[..., :Tc]
                    ratio_c = (xhat_clean.norm() / (xb_clean.norm() + 1e-8)).item()
                    msg += f" | clean energy ratio={ratio_c:.3f}"
                print(msg)

        # Save best
        if running["total"] < best_loss - 1e-6:
            best_loss = running["total"]
            save_path = Path(cfg.out_dir) / "noise_vae_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "T": T_model,
                    "z_dim": cfg.z_dim,
                    "state_dict": vae.state_dict(),
                    "epoch": epoch,
                    "cfg": vars(cfg),
                },
                save_path,
            )
            print(f"  [saved] {save_path} (best so far: {best_loss:.5f})")

        # Save periodic
        if cfg.save_every and (epoch % cfg.save_every == 0):
            save_path = Path(cfg.out_dir) / f"noise_vae_epoch{epoch:03d}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "T": T_model,
                    "z_dim": cfg.z_dim,
                    "state_dict": vae.state_dict(),
                    "epoch": epoch,
                    "cfg": vars(cfg),
                },
                save_path,
            )
            print(f"  [saved] {save_path}")

    print("[done]")


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train NoiseVAE on interference with clean-suppression")
    p.add_argument("--noise_subj", type=str, required=True,
                   help="Path to subject dir with interference.npy")
    p.add_argument("--clean_subj", type=str, default=None,
                   help="Path to subject dir with clean.npy (optional but recommended)")
    p.add_argument("--channel", type=int, default=0, help="Channel index to use")
    p.add_argument("--z_dim", type=int, default=256)
    p.add_argument("--T_model", type=int, default=None,
                   help="If set, overrides computed T_model (must be multiple of 64)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    # Loss weights / schedules
    p.add_argument("--beta_final", type=float, default=1e-3, help="Final KL weight after warmup")
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--lambda_dc", type=float, default=1e-4)
    p.add_argument("--lambda_feat", type=float, default=1e-3, help="Early feature loss weight (after warmup)")
    p.add_argument("--feature_upto", type=int, default=2, help="Use encoder blocks [0..upto] for features")
    p.add_argument("--lambda_band", type=float, default=0.05, help="EEG-band penalty weight (0 to disable)")
    p.add_argument("--sfreq", type=float, default=512.0)
    p.add_argument("--band_penalize_low", type=float, default=1.0)
    p.add_argument("--band_penalize_high", type=float, default=20.0)
    p.add_argument("--lambda_clean", type=float, default=0.2, help="Clean-suppression weight (0 to disable)")

    # Misc
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=0)
    args = p.parse_args()

    # pack band tuple for convenience
    args.band_penalize = (args.band_penalize_low, args.band_penalize_high)
    return args


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
