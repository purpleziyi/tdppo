import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tdppo.datasets import load_sonarcloud_issues, build_monthly_snapshots
from tdppo.featurizer import featurize, build_dev_rank_map
from tdppo.env import TDEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# -------------------------
# 0) Configuration
# -------------------------
REPO = "aws_aws-sdk-java-v2"
MAX_ISSUES = 500

# Main training budgets used in the thesis:
# 5,000 / 15,000 / 50,000 timesteps
TOTAL_TIMESTEPS = 50000

# Number of recommendations generated per monthly snapshot.
MAX_RECOMMEND = 5

# Base directory for PMS logs.
# A separate subdirectory is created for each month
# so that progress.csv is not overwritten.
BASE_LOG_DIR = "logs_pms_aws"

os.makedirs(BASE_LOG_DIR, exist_ok=True)

# -------------------------
# 1) Load SonarCloud issues and construct monthly snapshots
# -------------------------
# CLOSED SonarCloud issues are used here as the practical static-analysis data source.
df = load_sonarcloud_issues(REPO, max_issues=MAX_ISSUES, status="CLOSED")

# Build monthly snapshots using the backlog state at the first day of each month.
snapshots = build_monthly_snapshots(df)

# Store monthly Top-K outputs and aggregated reward-curve data.
all_rows = []          # Consolidated monthly Top-K outputs → pms.csv
agg_reward_frames = [] # Aggregated progress.csv data across months

for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues in backlog)")
        continue

    # Reset the index at the beginning of each monthly loop
    # so that environment actions align with row positions.
    issues_df = issues_df.reset_index(drop=True)

    print(f"\n[PMS-RL] Training on snapshot {month}, backlog size={len(issues_df)}")

    # -------------------------
    # 2) Build practice reward (month-specific rank)
    #    Issues closed in the current month receive rank = 1..N → reward = 1/rank
    #    Issues not closed in the current month receive the small default penalty (-0.01)
    # -------------------------
    dev_rank_map = build_dev_rank_map(issues_df, month)

    # -------------------------
    # 3) Feature engineering (practice mode → TF-IDF + age)
    # -------------------------
    # Practice mode uses TF-IDF text features together with normalized age_days.
    X = featurize(issues_df, mode="practice")

    meta = {
        "dev_rank": dev_rank_map,                 # The environment converts this to 1.0 / rank
        "issue_ids": issues_df["issue_id"].tolist(),  # Preserved for result export
    }


    # -------------------------
    # 4) Build the environment and configure per-month logging
    # -------------------------
    def make_env():
        return TDEnv(X, meta, mode="practice", max_select=MAX_RECOMMEND)


    env = make_vec_env(make_env, n_envs=1)

    # Use a separate log directory for each month.
    month_tag = month.strftime("%Y-%m")
    month_log_dir = os.path.join(BASE_LOG_DIR, month_tag)
    os.makedirs(month_log_dir, exist_ok=True)

    # Generate {month}/progress.csv through the SB3 logger.
    month_logger = configure(month_log_dir, ["stdout", "csv"])

    # -------------------------
    # 5) Train PPO in practical static-analysis mode
    # -------------------------
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")  # or "cpu"
    model.set_logger(month_logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # -------------------------
    # 6) Evaluate greedy Top-K selections after training
    # -------------------------
    obs = env.reset()
    actions_buffer = []
    picked = set()  # Prevent duplicate selections in the exported Top-K list

    for _ in range(MAX_RECOMMEND):
        action, _ = model.predict(obs, deterministic=True)
        a0 = int(action[0]) if hasattr(action, "__len__") else int(action)

        if a0 in picked:
            # Fallback: choose the first not-yet-selected index
            # if the policy repeats an already selected issue.
            for cand in range(len(meta["issue_ids"])):
                if cand not in picked:
                    a0 = cand
                    break

        picked.add(a0)

        # Compute the practical reward associated with this action
        # using the same rule as in the environment: 1/rank or -0.01.
        # This value is exported only as a reference in the output CSV.
        rank = dev_rank_map.get(a0, None)
        reward_val = (1.0 / rank) if rank is not None else -0.01

        # Access the selected issue row by position.
        row = issues_df.iloc[a0]
        issue_id = meta["issue_ids"][a0]

        actions_buffer.append({
            "snapshot": month,
            "issue_idx": a0,
            "issue_id": issue_id,
            "reward_practice": reward_val,
            "created_at": row["created_at"],
            "closed_at": row["closed_at"],
            "severity": issues_df.loc[issues_df["issue_id"] == issue_id, "severity"].values[0],
            "effort_minutes": issues_df.loc[issues_df["issue_id"] == issue_id, "effort_minutes"].values[0],
        })

        # Advance the environment by executing the selected action.
        obs, _, _, _ = env.step([a0])

    # Sort the monthly Top-K list by practical reward in descending order.
    actions_sorted = sorted(actions_buffer, key=lambda x: x["reward_practice"], reverse=True)
    print(
        f"[PMS-RL] Snapshot {month}: Top-{MAX_RECOMMEND} →",
        [(r["issue_id"], round(r["reward_practice"], 4)) for r in actions_sorted]
    )
    all_rows.extend(actions_sorted)

    # -------------------------
    # 7) Read the monthly progress.csv and keep the key learning-curve columns
    # -------------------------
    progress_path = os.path.join(month_log_dir, "progress.csv")
    if os.path.exists(progress_path):
        prog = pd.read_csv(progress_path)
        if set(["time/total_timesteps", "rollout/ep_rew_mean"]).issubset(prog.columns):
            tmp = prog[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna().copy()
            tmp.rename(columns={
                "time/total_timesteps": "timesteps",
                "rollout/ep_rew_mean": "mean_reward"
            }, inplace=True)
            tmp["month"] = month_tag
            agg_reward_frames.append(tmp)
        else:
            print(f"⚠️ {progress_path} missing required columns.")
    else:
        print(f"⚠️ {progress_path} not found.")

# -------------------------
# 8) Save pms.csv (Top-K results across all months)
# -------------------------
if all_rows:
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv("pms.csv", index=False)
    print("\nSaved practice for SQ rankings to pms.csv")
else:
    print("\nNo practice results to save (all snapshots were empty or skipped).")

# -------------------------
# 9) Aggregate reward curves across months
# -------------------------
if agg_reward_frames:
    curve_df = pd.concat(agg_reward_frames, ignore_index=True)
    curve_df.to_csv("reward_curve_pms.csv", index=False)

    # Plot multiple learning curves, one for each monthly snapshot.
    plt.figure(figsize=(10,6))
    for m in curve_df["month"].unique():
        sub = curve_df[curve_df["month"] == m]
        plt.plot(sub["timesteps"], sub["mean_reward"], label=m)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")

    # Place the legend outside the plot area so that it does not cover the lines.
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
    plt.tight_layout()  # Automatically adjust spacing to avoid clipping
    plt.savefig("reward_curve_pms.png")
    print("Saved practice reward curve to reward_curve_pms.png")
else:
    print("No reward curves to aggregate (no progress.csv found).")





