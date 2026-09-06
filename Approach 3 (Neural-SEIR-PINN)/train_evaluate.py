"""
train_evaluate.py
==================
Full training and evaluation pipeline for the Neural SEIR PINN model.
Includes:
  - GPU-accelerated training with AdamW + Cosine LR schedule
  - Early stopping
  - Train/test split (80/20)
  - Out-of-sample evaluation: R2, MAE, RMSE, MSE, Adj-R2
  - Saves best model weights to disk
  - Computes and prints dynamic Rt estimates
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from neural_seir_pinn import NeuralSEIR_PINN, PhysicsInformedLoss, EpidemicDataset


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    "data_file": "multimodal_aligned_data.csv",
    "seq_len": 14,              # 14-day sliding window
    "batch_size": 32,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "lambda_physics": 0.05,
    "epochs": 200,
    "patience": 20,             # early stopping
    "train_ratio": 0.80,
    "output_dir": "Approach 3 (Neural-SEIR-PINN) artifacts",
    "model_checkpoint": "best_seir_pinn.pt",
}


# =============================================================================
# Helper functions
# =============================================================================
def get_feature_cols(df):
    """Select relevant feature columns from the merged multimodal dataset."""
    mobility_cols = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]
    base_cols = ["Confirmed_Daily", "Deaths_Daily", "Recovered_Daily", "StringencyIndex"]
    # Only include mobility cols that actually exist in the dataframe
    available_mob = [c for c in mobility_cols if c in df.columns]
    return base_cols + available_mob


def adjusted_r2(r2, n, k):
    if n - k - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_metrics(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    adj_r2 = adjusted_r2(r2, len(y_true), n_features)
    return {"R2": r2, "Adj_R2": adj_r2, "MAE": mae, "MSE": mse, "RMSE": rmse}


# =============================================================================
# Main Training & Evaluation Function
# =============================================================================
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load data
    print("\nLoading multimodal dataset...")
    df = pd.read_csv(CONFIG["data_file"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Total records: {len(df)} days ({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})")

    feature_cols = get_feature_cols(df)
    n_features = len(feature_cols)
    print(f"  Feature columns ({n_features}): {feature_cols}")

    # 2. Build Dataset
    dataset = EpidemicDataset(
        df, feature_cols,
        target_infected_col="Confirmed_Daily",
        target_recovered_col="Recovered_Daily",
        seq_len=CONFIG["seq_len"],
        scale=True
    )
    inf_scale  = dataset.inf_scale
    rec_scale  = dataset.rec_scale

    n_train = int(len(dataset) * CONFIG["train_ratio"])
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)
    print(f"  Train samples: {n_train}, Val samples: {n_val}")

    # 3. Build Model
    model = NeuralSEIR_PINN(
        n_features=n_features,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel has {total_params:,} trainable parameters.")

    loss_fn   = PhysicsInformedLoss(lambda_physics=CONFIG["lambda_physics"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    # 4. Training Loop with Early Stopping
    print(f"\nStarting training for up to {CONFIG['epochs']} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(os.path.join("..", CONFIG["output_dir"]), exist_ok=True)
    checkpoint_path = os.path.join("..", CONFIG["output_dir"], CONFIG["model_checkpoint"])

    for epoch in range(1, CONFIG["epochs"] + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for x_batch, y_inf, y_rec in train_loader:
            x_batch = x_batch.to(device)
            y_inf   = y_inf.to(device).unsqueeze(-1).expand(-1, CONFIG["seq_len"])
            y_rec   = y_rec.to(device).unsqueeze(-1).expand(-1, CONFIG["seq_len"])

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss, _ = loss_fn(outputs, y_inf, y_rec)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_inf, y_rec in val_loader:
                x_batch = x_batch.to(device)
                y_inf   = y_inf.to(device).unsqueeze(-1).expand(-1, CONFIG["seq_len"])
                y_rec   = y_rec.to(device).unsqueeze(-1).expand(-1, CONFIG["seq_len"])
                outputs = model(x_batch)
                loss, _ = loss_fn(outputs, y_inf, y_rec)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{CONFIG['epochs']} | "
                  f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                break

    print(f"\nTraining complete. Best model saved to: {checkpoint_path}")

    # 5. Load best model for evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 6. Collect full predictions (validation set)
    all_true_inf, all_pred_inf = [], []
    all_true_rec, all_pred_rec = [], []
    all_rt = []

    with torch.no_grad():
        for x_batch, y_inf, y_rec in val_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            pred_inf  = outputs["pred_infected"][:, -1].cpu().numpy()  # last step
            pred_rec  = outputs["pred_recovered"][:, -1].cpu().numpy()
            rt_vals   = outputs["Rt"][:, -1].cpu().numpy()
            all_pred_inf.extend(pred_inf)
            all_true_inf.extend(y_inf.numpy())
            all_pred_rec.extend(pred_rec)
            all_true_rec.extend(y_rec.numpy())
            all_rt.extend(rt_vals)

    # Rescale back to original counts
    all_pred_inf  = np.array(all_pred_inf)  * inf_scale
    all_true_inf  = np.array(all_true_inf)  * inf_scale
    all_pred_rec  = np.array(all_pred_rec)  * rec_scale
    all_true_rec  = np.array(all_true_rec)  * rec_scale
    all_rt        = np.array(all_rt)

    # 7. Compute Metrics
    metrics_inf = compute_metrics(all_true_inf, all_pred_inf, n_features)
    metrics_rec = compute_metrics(all_true_rec, all_pred_rec, n_features)

    print("\n" + "="*60)
    print("EVALUATION RESULTS (Out-of-Sample Validation Set)")
    print("="*60)
    print(f"\n  Confirmed Cases (Daily Infected):")
    for k, v in metrics_inf.items():
        print(f"    {k:12s}: {v:.4f}")
    print(f"\n  Recovery Cases (Daily Recovered):")
    for k, v in metrics_rec.items():
        print(f"    {k:12s}: {v:.4f}")
    print(f"\n  Effective Reproduction Number (Rt):")
    print(f"    Mean Rt  : {all_rt.mean():.4f}")
    print(f"    Rt range : [{all_rt.min():.4f} to {all_rt.max():.4f}]")

    # 8. Save results to CSV and markdown table
    results_df = pd.DataFrame([
        {"Target": "Confirmed Cases (Daily)", **metrics_inf},
        {"Target": "Recovered Cases (Daily)", **metrics_rec}
    ])

    results_csv = os.path.join("..", CONFIG["output_dir"], "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    results_md = os.path.join("..", CONFIG["output_dir"], "evaluation_results.md")
    with open(results_md, "w") as f:
        f.write("# Neural SEIR PINN — Evaluation Results (Out-of-Sample)\n\n")
        f.write(f"**Device used:** {device}  \n")
        if device.type == "cuda":
            f.write(f"**GPU:** {torch.cuda.get_device_name(0)}  \n")
        f.write(f"**Training records:** {n_train}  |  **Validation records:** {n_val}\n\n")
        f.write("## Model Performance Metrics\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write(f"\n\n## Effective Reproduction Number (Rt)\n\n")
        f.write(f"| Metric | Value |\n|:---|:---|\n")
        f.write(f"| Mean Rt | {all_rt.mean():.4f} |\n")
        f.write(f"| Min Rt  | {all_rt.min():.4f} |\n")
        f.write(f"| Max Rt  | {all_rt.max():.4f} |\n")
    print(f"\n  Results saved to:\n    {results_csv}\n    {results_md}")

    # 9. Save training history and raw predictions for plotting
    pd.DataFrame(history).to_csv(
        os.path.join("..", CONFIG["output_dir"], "training_history.csv"), index=False
    )
    pd.DataFrame({
        "true_infected":  all_true_inf,
        "pred_infected":  all_pred_inf,
        "true_recovered": all_true_rec,
        "pred_recovered": all_pred_rec,
        "Rt":             all_rt
    }).to_csv(os.path.join("..", CONFIG["output_dir"], "val_predictions.csv"), index=False)

    print("\nAll outputs saved. Run generate_visuals.py to create publication figures.")
    return history, results_df, all_rt


if __name__ == "__main__":
    train_and_evaluate()
