# lstm_train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Hyperparameters
LAYER = 2
HIDDEN = 128
DROPOUT = 0

class LSTMForecaster(nn.Module):
    def __init__(self, hist_feat=7, static_dim=4, hidden=HIDDEN, num_layers=LAYER, out_horizon=15, out_feat=7):
        super().__init__()
        self.in_dim = hist_feat + static_dim
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_horizon * out_feat),
        )
        self.out_horizon = out_horizon
        self.out_feat = out_feat

    def forward(self, x_hist, x_static):
        B, T, F = x_hist.shape  # (B,90,7)
        # tile static across time and concat to each step
        x_static_t = x_static.unsqueeze(1).expand(-1, T, -1)  # (B,90,4)
        x = torch.cat([x_hist, x_static_t], dim=-1)           # (B,90,11)
        h, _ = self.lstm(x)                                   # (B,90,H)
        last = h[:, -1, :]                                    # (B,H)
        out = self.head(last)                                 # (B,15*7)
        return out.view(B, self.out_horizon, self.out_feat)   # (B,15,7)

def _run_epoch(model, loader, crit, opt, device, train=True, pbar=False):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    count = 0
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
            total += loss.item() * xh.size(0)
            count += xh.size(0)
    return total / max(count, 1)

def train(train_loader, val_loader, config, progress_bar=True, early_stopping=True):
    device = torch.device(config.get("device", "cpu"))
    lr = config.get("optim", {}).get("lr", 1e-3)
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 7))

    # shapes
    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]

    model = LSTMForecaster(hist_feat=hist_F, static_dim=4, hidden=HIDDEN, num_layers=LAYER,
                           out_horizon=y_T, out_feat=y_F).to(device)
    opt = Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "best_val_loss": None}

    for epoch in range(1, epochs + 1):
        if progress_bar:
            print(f"Epoch {epoch}/{epochs}")
        tr_loss = _run_epoch(model, train_loader, crit, opt, device, train=True, pbar=progress_bar)
        va_loss = _run_epoch(model, val_loader,   crit, opt, device, train=False, pbar=progress_bar)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if progress_bar:
            print(f"  train MSE: {tr_loss:.6f} | val MSE: {va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stopping and bad_epochs >= patience:
                if progress_bar:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    history["best_val_loss"] = best_val
    return best_state if best_state is not None else model.state_dict(), history

def create_model(config):
    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]
    return LSTMForecaster(hist_feat=hist_F, static_dim=4, hidden=HIDDEN, num_layers=LAYER,
                          out_horizon=y_T, out_feat=y_F)

