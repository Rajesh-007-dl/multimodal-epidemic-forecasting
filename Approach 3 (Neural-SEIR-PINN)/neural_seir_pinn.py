"""
neural_seir_pinn.py
====================
Physics-Informed Neural Network (PINN) combining:
  - A Bidirectional LSTM to learn temporal dynamics from multi-modal inputs
  - A differentiable SEIR ODE layer that enforces epidemic transmission physics
  - A composite loss: Data MSE + Physics Constraint residuals

Architecture:
  Inputs: [Confirmed_Daily, Deaths_Daily, Recovered_Daily,
           Mobility_Features (x6), StringencyIndex]  => 10 features per time step

  Bi-LSTM -> LayerNorm -> Dropout -> Linear heads:
    - beta_t  : time-varying transmission rate (clamped > 0)
    - gamma_t : recovery rate (clamped > 0)
    - sigma_t : incubation rate (fixed prior or learned)
    - S, E, I, R compartment outputs (via SEIR ODE integration)

Loss:
    L_total = L_data + lambda_physics * L_physics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 1. Differentiable SEIR ODE Layer (Euler integration)
# =============================================================================
class SEIRLayer(nn.Module):
    """
    Integrates the SEIR ODE system forward one timestep using Euler method.
    All operations are differentiable through autograd.
    
    SEIR Equations:
        dS/dt = -beta * S * I / N
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - gamma * I
        dR/dt =  gamma * I
    """
    def __init__(self, N: float = 7_800_000_000.0):
        super().__init__()
        self.N = N  # World population as of 2020

    def forward(self, S, E, I, R, beta, gamma, sigma, dt=1.0):
        """
        Args:
            S, E, I, R: compartment sizes [batch]
            beta, gamma, sigma: parameters [batch]
            dt: time step in days
        Returns:
            S_new, E_new, I_new, R_new
        """
        N = self.N
        new_infections = beta * S * I / N
        new_exposed = sigma * E
        new_recovered = gamma * I

        dS = -new_infections
        dE = new_infections - new_exposed
        dI = new_exposed - new_recovered
        dR = new_recovered

        S_new = torch.clamp(S + dt * dS, min=0.0)
        E_new = torch.clamp(E + dt * dE, min=0.0)
        I_new = torch.clamp(I + dt * dI, min=0.0)
        R_new = torch.clamp(R + dt * dR, min=0.0)

        return S_new, E_new, I_new, R_new


# =============================================================================
# 2. Neural SEIR PINN Model
# =============================================================================
class NeuralSEIR_PINN(nn.Module):
    """
    Bidirectional LSTM + SEIR physics layer.
    
    Input: (batch, seq_len, n_features)
    Output: dict with predicted I(t), R(t), beta(t), gamma(t), Rt(t),
            and SEIR compartment trajectories.
    """
    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        population: float = 7_800_000_000.0,
        sigma_prior: float = 1.0 / 5.2   # mean incubation 5.2 days (COVID-19)
    ):
        super().__init__()
        self.population = population
        self.sigma_prior = sigma_prior

        # --- Bi-LSTM backbone ---
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # --- Parameter projection heads ---
        # beta: transmission rate (must be positive)
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()   # ensures beta > 0
        )
        # gamma: recovery rate (must be positive)
        self.gamma_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # --- Direct regression heads for I(t) and R(t) ---
        self.infected_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()   # counts must be >= 0
        )
        self.recovered_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        # --- SEIR physics layer ---
        self.seir = SEIRLayer(N=population)

    def forward(self, x, S0=None, E0=None, I0=None, R0=None):
        """
        Args:
            x: Input tensor [batch, seq_len, n_features]
            S0, E0, I0, R0: Optional initial compartment values [batch]
        Returns:
            dict with predicted infected, recovered, beta, gamma, Rt, 
            and physics-driven compartment trajectories
        """
        batch_size, seq_len, _ = x.shape
        N = self.population

        # Default initial conditions (Day 0 ~ pandemic start)
        device = x.device
        if S0 is None: S0 = torch.full((batch_size,), N - 1.0, device=device)
        if E0 is None: E0 = torch.zeros(batch_size, device=device)
        if I0 is None: I0 = torch.ones(batch_size, device=device)
        if R0 is None: R0 = torch.zeros(batch_size, device=device)

        # Bi-LSTM forward pass
        lstm_out, _ = self.bilstm(x)          # [B, T, H*2]
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Project to parameters at every timestep
        beta_seq  = self.beta_head(lstm_out).squeeze(-1)    # [B, T]
        gamma_seq = self.gamma_head(lstm_out).squeeze(-1)   # [B, T]
        sigma     = torch.tensor(self.sigma_prior, device=device)

        # Direct regression output
        pred_infected  = self.infected_head(lstm_out).squeeze(-1)   # [B, T]
        pred_recovered = self.recovered_head(lstm_out).squeeze(-1)  # [B, T]

        # Unroll SEIR ODE across the sequence
        S, E, I, R = S0.clone(), E0.clone(), I0.clone(), R0.clone()
        seir_I_traj = []
        seir_R_traj = []
        seir_S_traj = []
        seir_E_traj = []

        for t in range(seq_len):
            beta_t  = beta_seq[:, t]
            gamma_t = gamma_seq[:, t]
            S, E, I, R = self.seir(S, E, I, R, beta_t, gamma_t, sigma)
            seir_I_traj.append(I)
            seir_R_traj.append(R)
            seir_S_traj.append(S)
            seir_E_traj.append(E)

        seir_I = torch.stack(seir_I_traj, dim=1)  # [B, T]
        seir_R = torch.stack(seir_R_traj, dim=1)
        seir_S = torch.stack(seir_S_traj, dim=1)
        seir_E = torch.stack(seir_E_traj, dim=1)

        # Effective reproduction number Rt = beta / gamma
        Rt = beta_seq / (gamma_seq + 1e-8)

        return {
            "pred_infected":  pred_infected,   # Direct LSTM prediction
            "pred_recovered": pred_recovered,
            "seir_I":  seir_I,                 # Physics-driven SEIR I(t)
            "seir_R":  seir_R,
            "seir_S":  seir_S,
            "seir_E":  seir_E,
            "beta":    beta_seq,
            "gamma":   gamma_seq,
            "Rt":      Rt
        }


# =============================================================================
# 3. Physics-Informed Composite Loss
# =============================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Total Loss = L_data + lambda_physics * L_physics
    
    L_data    : MSE between LSTM predictions and true Infected/Recovered counts
    L_physics : Enforces SEIR ODE residuals and mass conservation
    """
    def __init__(self, lambda_physics: float = 0.1, N: float = 7_800_000_000.0):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.N = N
        self.mse = nn.MSELoss()

    def forward(self, outputs, true_infected, true_recovered):
        # --- Data loss: LSTM prediction vs. actual ---
        l_infected  = self.mse(outputs["pred_infected"],  true_infected)
        l_recovered = self.mse(outputs["pred_recovered"], true_recovered)
        l_data = l_infected + l_recovered

        # --- Physics loss 1: SEIR trajectory should also match actuals ---
        l_seir_I = self.mse(outputs["seir_I"], true_infected)
        l_seir_R = self.mse(outputs["seir_R"], true_recovered)

        # --- Physics loss 2: Mass conservation S+E+I+R = N ---
        total_pop = (outputs["seir_S"] + outputs["seir_E"] +
                     outputs["seir_I"] + outputs["seir_R"])
        l_conservation = self.mse(total_pop, 
                                  torch.full_like(total_pop, self.N))

        # --- Physics loss 3: Rt should stay in realistic bounds [0.1, 5.0] ---
        Rt = outputs["Rt"]
        l_rt = torch.relu(Rt - 5.0).mean() + torch.relu(0.1 - Rt).mean()

        l_physics = l_seir_I + l_seir_R + l_conservation + l_rt

        total_loss = l_data + self.lambda_physics * l_physics
        return total_loss, {
            "l_data": l_data.item(),
            "l_physics": l_physics.item(),
            "l_seir_I": l_seir_I.item(),
            "l_conservation": l_conservation.item(),
            "l_rt": l_rt.item()
        }


# =============================================================================
# 4. Dataset & DataLoader Helper
# =============================================================================
class EpidemicDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset for epidemic time series.
    Each sample is a (seq_len, n_features) window with targets at t=seq_len.
    """
    def __init__(self, df, feature_cols, target_infected_col, target_recovered_col,
                 seq_len=14, scale=True):
        self.seq_len = seq_len
        self.scale = scale

        features = df[feature_cols].values.astype(np.float32)
        infected  = df[target_infected_col].values.astype(np.float32)
        recovered = df[target_recovered_col].values.astype(np.float32)

        if scale:
            self.feat_mean = features.mean(axis=0)
            self.feat_std  = features.std(axis=0) + 1e-8
            self.inf_scale  = infected.max() + 1e-8
            self.rec_scale  = recovered.max() + 1e-8
            features  = (features  - self.feat_mean) / self.feat_std
            infected  = infected  / self.inf_scale
            recovered = recovered / self.rec_scale

        self.features  = features
        self.infected  = infected
        self.recovered = recovered

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x   = self.features[idx : idx + self.seq_len]
        y_i = self.infected[idx + self.seq_len]
        y_r = self.recovered[idx + self.seq_len]
        return (
            torch.tensor(x,   dtype=torch.float32),
            torch.tensor(y_i, dtype=torch.float32),
            torch.tensor(y_r, dtype=torch.float32)
        )


if __name__ == "__main__":
    print("Neural SEIR PINN module loaded. Testing with random data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Quick shape sanity check
    B, T, F = 4, 14, 10
    x_dummy = torch.randn(B, T, F).to(device)

    model = NeuralSEIR_PINN(n_features=F).to(device)
    loss_fn = PhysicsInformedLoss()
    
    outputs = model(x_dummy)
    print(f"  pred_infected shape: {outputs['pred_infected'].shape}")
    print(f"  seir_I shape: {outputs['seir_I'].shape}")
    print(f"  Rt range: [{outputs['Rt'].min().item():.3f}, {outputs['Rt'].max().item():.3f}]")

    true_I = torch.rand(B, T).to(device)
    true_R = torch.rand(B, T).to(device)
    loss, breakdown = loss_fn(outputs, true_I, true_R)
    print(f"  Total loss: {loss.item():.4f}, breakdown: {breakdown}")
    print("Neural SEIR PINN module OK!")
