# Transformer_train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import math

# ===================== Hyperparameters =====================
LAYERS = 3          # Transformer layers
MODEL_DIM = 192     # d_model
FF_DIM = 512        # feedforward inner dim
HEADS = 4           # attention heads
DROPOUT = 0.1
LR = None           # override config["optim"]["lr"] if set

# ===================== Positional Encoding =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ===================== Model =====================
class TransEncForecaster(nn.Module):
    """
    Input:  x_hist (B,90,7), x_static (B,4) -> concat static per step => (B,90,11)
    Linear -> d_model, +PE -> TransformerEncoder
    Head takes last token to predict (15*7)
    """
    def __init__(self, hist_feat=7, static_dim=4, d_model=192, ff_dim=512,
                 nhead=4, num_layers=3, dropout=0.1, out_horizon=15, out_feat=7):
        super().__init__()
        in_dim = hist_feat + static_dim  # 11
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model, max_len=512)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_horizon * out_feat),
        )
        self.out_h = out_horizon
        self.out_f = out_feat
        # init
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x_hist, x_static):
        B, T, F = x_hist.shape
        xs = x_static.unsqueeze(1).expand(-1, T, -1)     # (B,90,4)
        x = torch.cat([x_hist, xs], dim=-1)              # (B,90,11)
        x = self.proj(x)                                  # (B,90,d_model)
        x = self.pe(x)
        h = self.encoder(x)                               # (B,90,d_model)
        last = h[:, -1, :]                                # (B,d_model)
        out = self.head(last)                             # (B,15*7)
        return out.view(B, self.out_h, self.out_f)

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
    device = torch.device(config.get("device", "cpu"))
    lr_cfg = config.get("optim", {}).get("lr", 1e-3)
    lr = LR if LR is not None else lr_cfg
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 7))

    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]

    model = TransEncForecaster(
        hist_feat=hist_F, static_dim=4,
        d_model=MODEL_DIM, ff_dim=FF_DIM, nhead=HEADS, num_layers=LAYERS,
        dropout=DROPOUT, out_horizon=y_T, out_feat=y_F
    ).to(device)

    opt = Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "best_val_loss": None}

    for ep in range(1, epochs + 1):
        if progress_bar: print(f"Epoch {ep}/{epochs}")
        tr = _run_epoch(model, train_loader, crit, opt, device, True, progress_bar)
        va = _run_epoch(model, val_loader,   crit, opt, device, False, progress_bar)
        history["train_loss"].append(tr); history["val_loss"].append(va)
        if progress_bar: print(f"  train MSE: {tr:.6f} | val MSE: {va:.6f}")
        if va < best_val - 1e-6:
            best_val, best_state, bad = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if early_stopping and bad >= patience:
                if progress_bar: print(f"Early stopping at epoch {ep}.")
                break

    history["best_val_loss"] = best_val
    return best_state if best_state is not None else model.state_dict(), history

def create_model(config):
    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]
    return TransEncForecaster(
        hist_feat=hist_F, static_dim=4,
        d_model=MODEL_DIM, ff_dim=FF_DIM, nhead=HEADS, num_layers=LAYERS,
        dropout=DROPOUT, out_horizon=y_T, out_feat=y_F
    )
