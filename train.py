"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# Mission 1: three impossible readings appear in this log (found by plotting each column):
#   * battery_v == 0.0     -> a powered doorbell cannot measure 0 V; this is a fault / no-read.
#   * temp_c  <= -40       -> the only value here is -127.0, a classic sensor fault sentinel.
#   * wifi_rssi_dbm >= 0   -> RSSI is a negative dBm figure; 0 means "no reading", not a signal.
# These are physically impossible, so dropping them loses no real information. Real extremes
# (a 3.0 V battery, an 8 C room) are kept because a real sensor can genuinely report them.
bad = (df.battery_v <= 0) | (df.temp_c <= -40) | (df.wifi_rssi_dbm >= 0)
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# Mission 2: normalise with statistics from the TRAINING rows only. If we used all rows
# (including validation) we would leak information about the validation set into training and
# report an optimistic MAE. sd is guarded against zero so a constant column cannot divide by 0.
mu = X.mean(axis=0).astype(np.float32)
sd = X.std(axis=0).astype(np.float32)
sd[sd == 0] = 1.0

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")

# Mission 1: straight-line baseline. Fit days_left ~ [features, 1] by least squares, then
# score the same fit on the validation rows. A single straight line cannot capture the
# non-linear relationships (e.g. battery voltage vs days), so it should beat the mean but
# fall well short of the network.
A = np.hstack([X, np.ones((len(X), 1), np.float32)])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
yv_lin = np.hstack([Xv, np.ones((len(Xv), 1), np.float32)]) @ coef
print(f"baseline (linear regression): {mae(yv_lin, yv):.2f} days")

# Plot predicted against true to see where the straight line is most wrong (saved for the report).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.scatter(yv, yv_lin, s=10, alpha=0.5)
lo, hi = float(min(yv.min(), yv_lin.min())), float(max(yv.max(), yv_lin.max()))
plt.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect")
plt.xlabel("true days_left")
plt.ylabel("predicted days_left")
plt.title("Linear baseline: predicted vs true")
plt.legend()
plt.tight_layout()
plt.savefig("linear_pred_vs_true.png", dpi=120)
plt.close()

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}:            {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
