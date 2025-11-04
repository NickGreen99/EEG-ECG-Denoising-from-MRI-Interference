import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resunet import DeepDSP_UNetRes
from datasets import EEGDenoiseDataset

import numpy as np


def train(
    root_dir="data_segmented_91_10_TR_all_channels/",
    batch_size=32,
    epochs=250000,
    lr=3e-5,
    log_dir="runs/denoise",
    val_split=0.1,
    model_save_path="10_TR_unet_ecg.pt",
    use_adjacent=True,          # match your dataset/model input
    channels="single",          # "single" or "all"
    main_channel=0,             # used if channels="single"
    num_workers=2,
    monitor="real",             # "real" | "syn"  (which val to use for checkpointing)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)

    # --- subject selection (same as your earlier script) ---
    root = Path(root_dir)
    subj_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    random.shuffle(subj_dirs)

    n_val = max(1, int(val_split * len(subj_dirs)))
    train_subj = subj_dirs[0]  # you can change this if you want a different split
    print(f"Training on:  {train_subj}")
    print(f"Validating on: {train_subj} (both synthetic & real)")

    # --- datasets & loaders ---
    train_ds = EEGDenoiseDataset(
        train_subj,
        use_adjacent=use_adjacent,
        split_ratio=0.9,
        mode="train",
        seed=0,
        channels=channels,
        main_channel=main_channel,
    )
    val_syn_ds = EEGDenoiseDataset(
        train_subj,
        use_adjacent=use_adjacent,
        split_ratio=0.9,
        mode="val_syn",
        seed=0,
        channels=channels,
        main_channel=main_channel,
    )
    val_real_ds = EEGDenoiseDataset(
        train_subj,
        use_adjacent=use_adjacent,
        split_ratio=0.9,
        mode="val_real",
        seed=0,
        channels=channels,
        main_channel=main_channel,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_syn_dl = DataLoader(
        val_syn_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_real_dl = DataLoader(
        val_real_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # --- model / loss / optim ---
    in_channels = 1 + (23 if use_adjacent else 0)
    model = DeepDSP_UNetRes(in_channels=in_channels, out_channels=1, nb=6).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- logging ---
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"all_sensors"#f"dualval_adj{int(use_adjacent)}_{channels}_ch{main_channel}_long"
    writer = SummaryWriter(Path(log_dir) / ts / run_name)

    best_val = float("inf")
    global_step = 0

    def eval_loader(dloader):
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y, *_ in dloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred_noise = model(x)
                cleaned = x[:, 0:1, :] - pred_noise
                loss = criterion(cleaned, y)
                total += float(loss.item())
                n += 1
        return total / max(1, n)

    for epoch in range(1, epochs + 1):
        # shuffle noise permutation inside dataset each epoch
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

        # --------- TRAIN ---------
        model.train()
        running = 0.0
        for batch_idx, (x, y, *_) in enumerate(train_dl, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_noise = model(x)
            cleaned = x[:, 0:1, :] - pred_noise
            loss = criterion(cleaned, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            writer.add_scalar("Loss/train_batch", float(loss.item()), global_step)
            global_step += 1

        train_epoch_loss = running / max(1, len(train_dl))

        # --------- VALIDATION (both) ---------
        val_syn_loss = eval_loader(val_syn_dl)
        val_real_loss = eval_loader(val_real_dl)

        # choose monitor
        to_monitor = val_real_loss if monitor == "real" else val_syn_loss

        # logging
        writer.add_scalar("Loss/train_epoch", train_epoch_loss, epoch)
        writer.add_scalar("Loss/val_syn_epoch", val_syn_loss, epoch)
        writer.add_scalar("Loss/val_real_epoch", val_real_loss, epoch)

        print(f"[{epoch:04d}/{epochs}] "
              f"train={train_epoch_loss:.6f}  "
              f"val_syn={val_syn_loss:.6f}  "
              f"val_real={val_real_loss:.6f}")

        # checkpoint best
        if to_monitor < best_val:
            best_val = to_monitor
            torch.save(model.state_dict(), model_save_path)
            print(f"  ↳ Saved new best ({monitor})={best_val:.6f} to {model_save_path}")

    print(f"Training finished. Best {monitor} val: {best_val:.6f}")
    writer.close()


if __name__ == "__main__":
    train()
