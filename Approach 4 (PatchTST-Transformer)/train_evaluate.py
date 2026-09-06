"""
train_evaluate.py — PatchTST Temporal Transformer
====================================================
Full training and evaluation pipeline:
  - 56-day input window → 15-day direct multi-horizon forecast
  - GPU acceleration (RTX 4050)
  - AdamW + CosineAnnealingLR + Early stopping
  - Metrics: R², Adj-R², MAE, RMSE, MSE for Confirmed and Recovered
  - Saves best checkpoint and predictions for plotting
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from patchtst_model import PatchTST, PatchTSTDataset

# =============================================================================
# Config
# =============================================================================
CONFIG = {
    "data_file":        os.path.join("..", "Approach 3 (Neural-SEIR-PINN)", "multimodal_aligned_data.csv"),
    "seq_len":          56,
    "forecast_horizon": 15,
    "patch_size":       8,
    "patch_stride":     4,
    "d_model":          128,
    "n_heads":          8,
    "n_layers":         3,
    "d_ff":             256,
    "dropout":          0.2,
    "batch_size":       32,
    "lr":               5e-4,
    "weight_decay":     1e-4,
    "epochs":           250,
    "patience":         25,
    "train_ratio":      0.80,
    "output_dir":       os.path.join("..", "Approach 4 (PatchTST-Transformer) artifacts"),
    "checkpoint":       "best_patchtst.pt",
}

FEATURE_COLS = [
    "Confirmed_Daily", "Deaths_Daily", "Recovered_Daily", "StringencyIndex",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
]
TARGET_COLS = ["Confirmed_Daily", "Recovered_Daily"]


def adjusted_r2(r2, n, k):
    if n - k - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_metrics(y_true, y_pred, k):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {"R2": r2, "Adj_R2": adjusted_r2(r2, len(y_true), k),
            "MAE": mae, "MSE": mse, "RMSE": rmse}


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # -------------------------------------------------------------------------
    # Load & validate data
    # -------------------------------------------------------------------------
    print("\nLoading multimodal dataset...")
    df = pd.read_csv(CONFIG["data_file"]).sort_values("Date").reset_index(drop=True)
    print(f"  Total records: {len(df)} days ({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})")

    # Use only available feature columns
    avail_feats = [c for c in FEATURE_COLS if c in df.columns]
    n_channels  = len(avail_feats)
    print(f"  Features ({n_channels}): {avail_feats}")

    dataset = PatchTSTDataset(
        df, avail_feats, TARGET_COLS,
        seq_len=CONFIG["seq_len"],
        forecast_horizon=CONFIG["forecast_horizon"],
        scale=True
    )
    tgt_scale = dataset.tgt_scale
    print(f"  Dataset samples: {len(dataset)}")

    n_train = int(len(dataset) * CONFIG["train_ratio"])
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    print(f"  Train: {n_train}  |  Val: {n_val}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = PatchTST(
        n_channels=n_channels,
        seq_len=CONFIG["seq_len"],
        patch_size=CONFIG["patch_size"],
        patch_stride=CONFIG["patch_stride"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_targets=len(TARGET_COLS)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPatchTST has {total_params:,} trainable parameters.")
    print(f"  Patches per channel: {model.num_patches} (patch_size={CONFIG['patch_size']}, stride={CONFIG['patch_stride']})")

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"\nStarting training ({CONFIG['epochs']} epochs max, patience={CONFIG['patience']})...")
    best_val, patience_ctr = float("inf"), 0
    history = {"train_loss": [], "val_loss": []}
    ckpt_path = os.path.join(CONFIG["output_dir"], CONFIG["checkpoint"])

    for epoch in range(1, CONFIG["epochs"] + 1):
        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(loss_fn(pred, y).item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{CONFIG['epochs']} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val:
            best_val, patience_ctr = avg_val, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val:.6f}")
                break

    print(f"\nBest model saved -> {ckpt_path}")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())

    # Flatten: [N, H, 2] → pick final day of horizon for point-metrics
    all_pred = np.concatenate(all_pred, axis=0)   # [N, H, 2]
    all_true = np.concatenate(all_true, axis=0)   # [N, H, 2]

    # Rescale
    all_pred_sc = all_pred * tgt_scale[np.newaxis, np.newaxis, :]
    all_true_sc = all_true * tgt_scale[np.newaxis, np.newaxis, :]

    # Use horizon midpoint (day 7 of 15) for scalar metrics
    mid = CONFIG["forecast_horizon"] // 2
    metrics_conf = compute_metrics(all_true_sc[:, mid, 0], all_pred_sc[:, mid, 0], n_channels)
    metrics_rec  = compute_metrics(all_true_sc[:, mid, 1], all_pred_sc[:, mid, 1], n_channels)

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS — Day {mid+1} of {CONFIG['forecast_horizon']}-Day Horizon")
    print("="*60)
    for label, m in [("Daily Confirmed Cases", metrics_conf), ("Daily Recoveries", metrics_rec)]:
        print(f"\n  {label}:")
        for k, v in m.items():
            print(f"    {k:12s}: {v:.4f}")

    # Save results
    results_df = pd.DataFrame([
        {"Target": "Daily Confirmed Cases", **metrics_conf},
        {"Target": "Daily Recoveries",      **metrics_rec}
    ])
    md_path = os.path.join(CONFIG["output_dir"], "evaluation_results.md")
    with open(md_path, "w") as f:
        f.write("# PatchTST Temporal Transformer — Evaluation Results\n\n")
        f.write(f"**Device:** {device}  \n")
        if device.type == "cuda":
            f.write(f"**GPU:** {torch.cuda.get_device_name(0)}  \n")
        f.write(f"**Input window:** {CONFIG['seq_len']} days  |  "
                f"**Forecast horizon:** {CONFIG['forecast_horizon']} days\n\n")
        f.write("## Model Performance (Day 7 of 15-Day Horizon)\n\n")
        f.write(results_df.to_markdown(index=False))

    pd.DataFrame(history).to_csv(
        os.path.join(CONFIG["output_dir"], "training_history.csv"), index=False)

    # Save full horizon predictions for plotting
    np.save(os.path.join(CONFIG["output_dir"], "pred_all.npy"), all_pred_sc)
    np.save(os.path.join(CONFIG["output_dir"], "true_all.npy"), all_true_sc)

    print(f"\nResults saved to {CONFIG['output_dir']}")
    print("Run generate_visuals.py for publication figures.")

    return history, results_df


if __name__ == "__main__":
    train_and_evaluate()
