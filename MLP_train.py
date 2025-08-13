# MLP_train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# ===================== Hyperparameters =====================
DEPTH = 4          # number of hidden layers
WIDTH = 512        # hidden width
DROPOUT = 0.1      # dropout between hidden layers
LR = None          # override config["optim"]["lr"] if not None
ACTIVATION = "relu"  # "relu" or "gelu"

def _act(name):
    return nn.ReLU() if name.lower() == "relu" else nn.GELU()

# ===================== Model =====================
class MLPForecaster(nn.Module):
    """
    Input options:
      1) flat: (B, 90*7+4)  -> call as model(x_flat)
      2) hist+static: x_hist (B,90,7), x_static (B,4) -> call as model(x_hist, x_static)
    Output: (B, 15, 7) scaled
    """
    def __init__(self, flat_dim, out_horizon=15, out_feat=7, width=512, depth=4, dropout=0.1, activation="relu"):
        super().__init__()
        layers = []
        in_dim = flat_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), _act(activation), nn.Dropout(dropout)]
            in_dim = width
        layers += [nn.Linear(in_dim, out_horizon * out_feat)]
        self.net = nn.Sequential(*layers)
        self.out_h = out_horizon
        self.out_f = out_feat
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_flat_or_hist, x_static=None):
        """
        Backward-compatible forward:
          - If x_static is None, treat first arg as flat vector.
          - Else, flatten hist and concat static.
        This preserves state_dict keys so old checkpoints still load.
        """
        if x_static is None:
            flat = x_flat_or_hist                      # (B, 90*7+4)
        else:
            x_hist = x_flat_or_hist                    # (B,90,7)
            B = x_hist.size(0)
            flat = torch.cat([x_hist.reshape(B, -1), x_static], dim=-1)  # (B, 90*7+4)
        out = self.net(flat)                           # (B, 15*7)
        return out.view(-1, self.out_h, self.out_f)    # (B,15,7)

# ===================== Train/Eval =====================
def _run_epoch(model, loader, crit, opt, device, train=True, pbar=False):
    model.train() if train else model.eval()
    total, count = 0.0, 0
    itr = tqdm(loader, leave=False) if pbar else loader
    with torch.set_grad_enabled(train):
        for batch in itr:
            # keep training as flat input (fast & simple)
            xf = batch["flat"].to(device)
            y  = batch["target"].to(device)
            yhat = model(xf)  # <-- passes only flat
            loss = crit(yhat, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = xf.size(0)
            total += loss.item() * bs
            count += bs
    return total / max(count, 1)

def train(train_loader, val_loader, config, progress_bar=True, early_stopping=True):
    device = torch.device(config.get("device", "cpu"))
    lr_cfg = config.get("optim", {}).get("lr", 1e-3)
    lr = LR if LR is not None else lr_cfg
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 7))

    flat_dim = int(config["x_flat_dim"])
    y_T, y_F = config["y_shape"]

    model = MLPForecaster(flat_dim, out_horizon=y_T, out_feat=y_F,
                          width=WIDTH, depth=DEPTH, dropout=DROPOUT, activation=ACTIVATION).to(device)
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
    """
    For mainTest.py: returns a model that can be called as model(x_hist, x_static)
    or model(x_flat). Using MLPForecaster directly keeps state_dict keys unchanged.
    """
    flat_dim = int(config["x_flat_dim"])
    y_T, y_F = config["y_shape"]
    return MLPForecaster(flat_dim, out_horizon=y_T, out_feat=y_F,
                         width=WIDTH, depth=DEPTH, dropout=DROPOUT, activation=ACTIVATION)
