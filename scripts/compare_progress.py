# compare_progress.py
# Compare two SB3 PPO progress logs under logs/ directory

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Config ----------
LOG_DIR = "logs"
FILE_1500 = "progress_aws 1500issues 50000steps.csv"   # with + beta*age_score
FILE_3700 = "progress_aws 3700issues 50000steps.csv"   # without age_score
TAIL_N = 50            # stats computed over the last N rows (auto-clamped if file shorter)
ROLLING_WINDOW = 10    # for plotting smooth line (visual only, data不舍弃)
OUT_SUMMARY = os.path.join(LOG_DIR, "progress_compare_summary.csv")
OUT_FIG = os.path.join(LOG_DIR, "progress_compare_plots.png")

# ---------- Helpers ----------
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    if "time/total_timesteps" not in df.columns:
        raise ValueError(f"{path} missing column: time/total_timesteps")
    return df

def tail_stats(series: pd.Series, n: int):
    n = min(n, len(series))
    tail = series.tail(n).dropna()
    if tail.empty:
        return np.nan, np.nan, np.nan
    return float(tail.mean()), float(tail.std(ddof=0)), float(series.dropna().iloc[-1])

def pick(df: pd.DataFrame, col: str):
    """Return column if exists else a NaN series matching index."""
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan]*len(df), index=df.index)

def clamp_tail_n(df: pd.DataFrame, n: int) -> int:
    return min(n, max(1, len(df)))

# ---------- Load ----------
path_1500 = os.path.join(LOG_DIR, FILE_1500)
path_3700 = os.path.join(LOG_DIR, FILE_3700)

df_1500 = load_csv(path_1500)
df_3700 = load_csv(path_3700)

# ---------- Metrics to compare ----------
metric_map = {
    "rollout/ep_rew_mean": "Reward (ep_rew_mean)",
    "train/entropy_loss": "Entropy loss",
    "train/explained_variance": "Explained variance",
    "train/value_loss": "Value loss",
    # 也可加： "train/approx_kl": "Approx KL",
}

# ---------- Prepare summary ----------
rows = []
tail_n_1500 = clamp_tail_n(df_1500, TAIL_N)
tail_n_3700 = clamp_tail_n(df_3700, TAIL_N)

for col, nice in metric_map.items():
    m1500 = pick(df_1500, col)
    m3700 = pick(df_3700, col)

    mean1500, std1500, last1500 = tail_stats(m1500, tail_n_1500)
    mean3700, std3700, last3700 = tail_stats(m3700, tail_n_3700)

    rows.append({
        "metric": nice,
        "col_name": col,
        "1500_mean_lastN": mean1500,
        "1500_std_lastN": std1500,
        "1500_last": last1500,
        "3700_mean_lastN": mean3700,
        "3700_std_lastN": std3700,
        "3700_last": last3700,
        "lastN_used_1500": tail_n_1500,
        "lastN_used_3700": tail_n_3700,
    })

summary = pd.DataFrame(rows, columns=[
    "metric", "col_name",
    "1500_mean_lastN", "1500_std_lastN", "1500_last",
    "3700_mean_lastN", "3700_std_lastN", "3700_last",
    "lastN_used_1500", "lastN_used_3700"
])

# Save summary CSV
summary.to_csv(OUT_SUMMARY, index=False)
print(f"Saved summary: {OUT_SUMMARY}")
print(summary.to_string(index=False))

# ---------- Plot ----------
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(13, 8))

# (1) Reward curves
ax1 = plt.subplot(2, 2, 1)
t1500 = df_1500["time/total_timesteps"]
t3700 = df_3700["time/total_timesteps"]
r1500 = pick(df_1500, "rollout/ep_rew_mean")
r3700 = pick(df_3700, "rollout/ep_rew_mean")

ax1.plot(t1500, r1500, label="1500 (+age)", linewidth=1.0, alpha=0.6)
ax1.plot(t3700, r3700, label="3700 (no age)", linewidth=1.0, alpha=0.6)

# overlay rolling mean (visual smoothing, 不丢数据)
if ROLLING_WINDOW and ROLLING_WINDOW > 1:
    ax1.plot(t1500, r1500.rolling(ROLLING_WINDOW).mean(), label="1500 (+age) rolling", linewidth=2.0, alpha=0.9)
    ax1.plot(t3700, r3700.rolling(ROLLING_WINDOW).mean(), label="3700 (no age) rolling", linewidth=2.0, alpha=0.9)

ax1.set_title("Reward curve comparison")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Mean episode reward")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# (2) Entropy loss
ax2 = plt.subplot(2, 2, 2, sharex=ax1)
e1500 = pick(df_1500, "train/entropy_loss")
e3700 = pick(df_3700, "train/entropy_loss")
ax2.plot(t1500, e1500, label="1500 (+age)", linewidth=1.0, alpha=0.6)
ax2.plot(t3700, e3700, label="3700 (no age)", linewidth=1.0, alpha=0.6)
if ROLLING_WINDOW and ROLLING_WINDOW > 1:
    ax2.plot(t1500, e1500.rolling(ROLLING_WINDOW).mean(), label="1500 (+age) rolling", linewidth=2.0, alpha=0.9)
    ax2.plot(t3700, e3700.rolling(ROLLING_WINDOW).mean(), label="3700 (no age) rolling", linewidth=2.0, alpha=0.9)
ax2.set_title("Entropy loss (lower → more deterministic)")
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Entropy loss")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# (3) Explained variance
ax3 = plt.subplot(2, 2, 3, sharex=ax1)
ev1500 = pick(df_1500, "train/explained_variance")
ev3700 = pick(df_3700, "train/explained_variance")
ax3.plot(t1500, ev1500, label="1500 (+age)", linewidth=1.0, alpha=0.6)
ax3.plot(t3700, ev3700, label="3700 (no age)", linewidth=1.0, alpha=0.6)
if ROLLING_WINDOW and ROLLING_WINDOW > 1:
    ax3.plot(t1500, ev1500.rolling(ROLLING_WINDOW).mean(), label="1500 (+age) rolling", linewidth=2.0, alpha=0.9)
    ax3.plot(t3700, ev3700.rolling(ROLLING_WINDOW).mean(), label="3700 (no age) rolling", linewidth=2.0, alpha=0.9)
ax3.set_title("Explained variance (higher is better)")
ax3.set_xlabel("Timesteps")
ax3.set_ylabel("Explained variance")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# (4) Value loss
ax4 = plt.subplot(2, 2, 4, sharex=ax1)
vl1500 = pick(df_1500, "train/value_loss")
vl3700 = pick(df_3700, "train/value_loss")
ax4.plot(t1500, vl1500, label="1500 (+age)", linewidth=1.0, alpha=0.6)
ax4.plot(t3700, vl3700, label="3700 (no age)", linewidth=1.0, alpha=0.6)
if ROLLING_WINDOW and ROLLING_WINDOW > 1:
    ax4.plot(t1500, vl1500.rolling(ROLLING_WINDOW).mean(), label="1500 (+age) rolling", linewidth=2.0, alpha=0.9)
    ax4.plot(t3700, vl3700.rolling(ROLLING_WINDOW).mean(), label="3700 (no age) rolling", linewidth=2.0, alpha=0.9)
ax4.set_title("Value loss (lower is better)")
ax4.set_xlabel("Timesteps")
ax4.set_ylabel("Value loss")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
print(f"Saved figure: {OUT_FIG}")

# ---------- Quick textual conclusion (optional) ----------
def safe_last_mean(df, col):
    s = pick(df, col)
    m = s.tail(clamp_tail_n(df, TAIL_N)).mean()
    return np.nan if pd.isna(m) else float(m)

rew1500 = safe_last_mean(df_1500, "rollout/ep_rew_mean")
rew3700 = safe_last_mean(df_3700, "rollout/ep_rew_mean")

print("\nQuick view (mean of last N):")
print(f"  Reward mean (1500 +age): {rew1500:.4f}")
print(f"  Reward mean (3700 no age): {rew3700:.4f}")
print("\nInterpretation:")
print("- Higher reward is better; lower value_loss and higher explained_variance are signs of a more stable critic.")
print("- Entropy typically decreases as the policy becomes more deterministic; extremely low entropy too early may indicate premature convergence.")
