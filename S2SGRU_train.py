# Seq2SeqGRU_train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random

# ===================== Hyperparameters =====================
LAYER = 2           # num_layers for encoder/decoder GRUs
HIDDEN = 128        # hidden size
DROPOUT = 0.1       # GRU dropout (applies when LAYER > 1)
LR = None           # if set, overrides config["optim"]["lr"]
TEACHER_FORCING = 0.5   # probability per time step to feed ground truth during training
CLIP_NORM = 1.0     # gradient clipping

# ===================== Model =====================
class Seq2SeqGRU(nn.Module):
    """
    Encoder:
        Input per step: concat(hist_feat(7), static(4)) -> (B,90,11)
        GRU -> final hidden h encodes the 90-day context.
    Decoder:
        Autoregressive for 15 steps.
        Input per step: concat(prev_y(7), static(4)) -> GRU -> Linear -> y_t(7)
        h_0^{dec} initialized from encoder h_N (optionally projected).

    forward(x_hist, x_static, target=None, teacher_forcing_ratio=0.0)
        - If target provided and ratio>0, uses teacher forcing during training.
        - Returns (B, 15, 7) in scaled space [0,1].
    """
    def __init__(self, hist_feat=7, static_dim=4, out_horizon=15, out_feat=7,
                 hidden=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hist_feat = hist_feat
        self.static_dim = static_dim
        self.out_h = out_horizon
        self.out_f = out_feat
        self.hidden = hidden
        self.num_layers = num_layers

        enc_in = hist_feat + static_dim   # 7 + 4 = 11
        dec_in = out_feat + static_dim    # 7 + 4 = 11

        enc_dropout = dropout if num_layers > 1 else 0.0
        dec_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.GRU(
            input_size=enc_in, hidden_size=hidden,
            num_layers=num_layers, batch_first=True, dropout=enc_dropout
        )
        self.init_dec = nn.Linear(hidden, hidden)  # allow a small transform on h_enc -> h_dec
        self.decoder = nn.GRU(
            input_size=dec_in, hidden_size=hidden,
            num_layers=num_layers, batch_first=True, dropout=dec_dropout
        )
        self.out_head = nn.Linear(hidden, out_feat)
        # learned start token for y_{0}
        self.start_token = nn.Parameter(torch.zeros(out_feat))

        # init
        nn.init.xavier_uniform_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        nn.init.xavier_uniform_(self.init_dec.weight)
        nn.init.zeros_(self.init_dec.bias)

    def forward(self, x_hist, x_static, target=None, teacher_forcing_ratio=0.0):
        """
        x_hist: (B,90,7) scaled
        x_static: (B,4)
        target: optional (B,15,7) scaled
        """
        B, T, F = x_hist.shape
        device = x_hist.device

        # ----- Encoder -----
        xs_rep = x_static.unsqueeze(1).expand(-1, T, -1)   # (B,90,4)
        enc_in = torch.cat([x_hist, xs_rep], dim=-1)       # (B,90,11)
        _, h_enc = self.encoder(enc_in)                    # h_enc: (num_layers, B, H)
        h_dec = torch.tanh(self.init_dec(h_enc))           # match shape (num_layers, B, H)

        # ----- Decoder (auto-regressive) -----
        outputs = []
        y_prev = self.start_token.unsqueeze(0).expand(B, -1)  # (B,7)

        for t in range(self.out_h):
            dec_step_in = torch.cat([y_prev, x_static], dim=-1).unsqueeze(1)  # (B,1,11)
            dec_out, h_dec = self.decoder(dec_step_in, h_dec)                 # dec_out: (B,1,H)
            y_t = self.out_head(dec_out.squeeze(1))                           # (B,7)
            outputs.append(y_t.unsqueeze(1))                                   # accumulate

            use_tf = (target is not None) and (random.random() < teacher_forcing_ratio)
            y_prev = target[:, t, :] if use_tf else y_t

        y_seq = torch.cat(outputs, dim=1)  # (B,15,7)
        return y_seq

# ===================== Train/Eval helpers =====================
def _run_epoch(model, loader, crit, opt, device, train=True, tf_prob=0.0, pbar=False):
    if train:
        model.train()
    else:
        model.eval()
    total, n = 0.0, 0
    itr = tqdm(loader, leave=False) if pbar else loader
    with torch.set_grad_enabled(train):
        for batch in itr:
            xh = batch["hist"].to(device)
            xs = batch["static"].to(device)
            y  = batch["target"].to(device)

            if train:
                yhat = model(xh, xs, target=y, teacher_forcing_ratio=tf_prob)
            else:
                yhat = model(xh, xs)  # autoregressive, no teacher forcing

            loss = crit(yhat, y)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if CLIP_NORM is not None and CLIP_NORM > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()

            bs = xh.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)

# ===================== Public API =====================
def train(train_loader, val_loader, config, progress_bar=True, early_stopping=True):
    """
    Compatible with mainTrain.py
    Returns:
        best_state_dict, history dict (train_loss, val_loss, best_val_loss)
    """
    device = torch.device(config.get("device", "cpu"))
    lr_cfg = config.get("optim", {}).get("lr", 1e-3)
    lr = LR if LR is not None else lr_cfg
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 7))

    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]

    model = Seq2SeqGRU(
        hist_feat=hist_F, static_dim=4,
        out_horizon=y_T, out_feat=y_F,
        hidden=HIDDEN, num_layers=LAYER, dropout=DROPOUT
    ).to(device)

    opt = Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train_loss": [], "val_loss": [], "best_val_loss": None}

    for ep in range(1, epochs + 1):
        if progress_bar:
            print(f"Epoch {ep}/{epochs}")

        tr = _run_epoch(model, train_loader, crit, opt, device,
                        train=True, tf_prob=TEACHER_FORCING, pbar=progress_bar)
        va = _run_epoch(model, val_loader,   crit, opt, device,
                        train=False, tf_prob=0.0, pbar=progress_bar)

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
                    print(f"Early stopping at epoch {ep} (no improvement {patience} epochs).")
                break

    history["best_val_loss"] = best_val
    return best_state if best_state is not None else model.state_dict(), history

def create_model(config):
    """Reconstructs the Seq2Seq model for evaluation (used by mainTest.py)."""
    hist_T, hist_F = config["x_hist_shape"]
    y_T, y_F = config["y_shape"]
    return Seq2SeqGRU(
        hist_feat=hist_F, static_dim=4,
        out_horizon=y_T, out_feat=y_F,
        hidden=HIDDEN, num_layers=LAYER, dropout=DROPOUT
    )
