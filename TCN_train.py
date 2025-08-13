# TCN_train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# ===================== Hyperparameters =====================
# You can tweak these directly.
LEVELS = 4           # number of temporal blocks (depth)
HIDDEN = 128         # channels in hidden layers
KERNEL_SIZE = 3      # conv kernel width
DROPOUT = 0.1        # dropout inside TCN blocks
WEIGHT_NORM = True   # apply weight_norm to Conv1d
LR = None            # if not None, overrides config["optim"]["lr"]

# ===================== TCN Building Blocks =====================

def _maybe_wn(conv):
    from torch.nn.utils import weight_norm as wn
    return wn(conv) if WEIGHT_NORM else conv

class Chomp1d(nn.Module):
    """Trims right-most 'chomp' timesteps to enforce causality after padding."""
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        # x: (B, C, T + chomp) -> (B, C, T)
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = _maybe_wn(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = _maybe_wn(nn.Conv1d(out_channels, out_channels, kernel_size,
                                         padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # residual projection if channels changed
        self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else nn.Identity()

        # init
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(self.resample, nn.Conv1d):
            nn.init.kaiming_uniform_(self.resample.weight, nonlinearity="linear")
            if self.resample.bias is not None:
                nn.init.zeros_(self.resample.bias)

    def forward(self, x):
        # x: (B, C_in, T)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = self.resample(x)
        return nn.functional.relu(out + res)

class TCN(nn.Module):
    """
    TCN stack with exponentially increasing dilations: 1, 2, 4, ...
    """
    def __init__(self, in_channels, hidden_channels, levels, kernel_size, dropout):
        super().__init__()
        layers = []
        prev_c = in_channels
        for i in range(levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(prev_c, hidden_channels, kernel_size, dilation, dropout)
            )
            prev_c = hidden_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C_in, T)
        return self.network(x)  # (B, C_hidden, T)

# ===================== Forecast Model =====================

class TCNForecaster(nn.Module):
    """
    Input:
      - x_hist: (B, 90, 7)  scaled
      - x_static: (B, 4)
      We broadcast static to each timestep -> (B, 90, 11),
      transpose to (B, 11, 90), run TCN, take last timestep features,
      and map to (15*7).
    Output:
      - (B, 15, 7) in scaled space [0,1]
    """
    def __init__(self, hist_feat=7, static_dim=4,
                 hidden=128, levels=4, kernel_size=3, dropout=0.1,
                 out_horizon=15, out_feat=7):
        super().__init__()
        in_channels = hist_feat + static_dim  # 11
        self.tcn = TCN(in_channels, hidden, levels, kernel_size, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_horizon * out_feat),
        )
        self.out_h = out_horizon
        self.out_f = out_feat

        # small init on linear layers
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_hist, x_static):
        B, T, F = x_hist.shape  # (B, 90, 7)
        xs = x_static.unsqueeze(1).expand(-1, T, -1)      # (B,90,4)
        x = torch.cat([x_hist, xs], dim=-1)               # (B,90,11)
        x = x.transpose(1, 2)                              # (B,11,90) for Conv1d
        h = self.tcn(x)                                    # (B,H,90)
        last = h[:, :, -1]                                 # (B,H) take last (causal)
        out = self.head(last)                              # (B, 15*7)
        return out.view(B, self.out_h, self.out_f)         # (B,15,7)

# ===================== Train/Eval =====================

def _run_epoch(model, loader, crit, opt, device, train=True, pbar=False):
    model.train() if train else model.eval()
    total, count = 0.0, 0
    itr = tqdm(loader, leave=False) if pbar else loader
    with torch.set_grad_enabled(train):
        for batch in itr:
            xh = batch["hist"].to(device)
            xs = batch["static"].to(device)
            y  = batch["target"].to(device)

            yhat = model(xh, xs)
            loss = crit(yhat, y)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            bs = xh.size(0)
            total += loss.item() * bs
            count += bs
    return total / max(count, 1)

def train(train_loader, val_loader, config, progress_bar=True, early_stopping=True):
    """
    Required by mainTrain.py:
      Returns best_state_dict and a history dict with train_loss, val_loss, best_val_loss.
    """
    device = torch.device(config.get("device", "cpu"))
    lr_cfg = config.get("optim", {}).get("lr", 1e-3)
    lr = LR if LR is not None else lr_cfg
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 7))

    # shapes from config
    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]

    model = TCNForecaster(
        hist_feat=hist_F, static_dim=4,
        hidden=HIDDEN, levels=LEVELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT,
        out_horizon=y_T, out_feat=y_F,
    ).to(device)

    opt = Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "best_val_loss": None}

    for ep in range(1, epochs + 1):
        if progress_bar:
            print(f"Epoch {ep}/{epochs}")
        tr = _run_epoch(model, train_loader, crit, opt, device, train=True,  pbar=progress_bar)
        va = _run_epoch(model, val_loader,   crit, opt, device, train=False, pbar=progress_bar)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if progress_bar:
            print(f"  train MSE: {tr:.6f} | val MSE: {va:.6f}")

        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if early_stopping and bad >= patience:
                if progress_bar:
                    print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs).")
                break

    history["best_val_loss"] = best_val
    return best_state if best_state is not None else model.state_dict(), history

# ===================== Factory for mainTest.py =====================

def create_model(config):
    """Rebuilds a TCNForecaster for evaluation (used by mainTest.py)."""
    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]
    return TCNForecaster(
        hist_feat=hist_F, static_dim=4,
        hidden=HIDDEN, levels=LEVELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT,
        out_horizon=y_T, out_feat=y_F,
    )
