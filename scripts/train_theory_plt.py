import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tdppo.datasets import load_sonarcloud_issues, build_monthly_snapshots_sonar
from tdppo.featurizer import featurize, map_severity, normalize
from tdppo.env import TDEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# --------------------------------------------------
# 1. Load SonarCloud issues and construct snapshots
# --------------------------------------------------
# Load up to 4,000 SonarCloud issues from the target project.
df = load_sonarcloud_issues("aws_aws-sdk-java-v2", max_issues=4000)

# Convert the issue stream into monthly snapshots.
snapshots = build_monthly_snapshots_sonar(df)

# Store Top-K results across all monthly snapshots.
all_results = []

# --------------------------------------------------
# Configure SB3 logging
# --------------------------------------------------
# Directory used to save SB3 training logs.
log_dir = "logs_tms_aws/"  # put logger into scripts/logs

# Log both to the terminal and to a CSV file (progress_tms_aws xx.csv).
new_logger = configure(log_dir, ["stdout", "csv"])

for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues)")
        continue

    print(f"Training on snapshot {month}, issues={len(issues_df)}")

    # --------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------
    # Build the theory-mode feature matrix used as the environment state.
    X = featurize(issues_df, mode="theory")

    # Map severity labels to numeric weights and normalize effort values.
    severity_score = map_severity(issues_df["severity"])
    effort_score = normalize(issues_df["effort_minutes"])

    # Age was previously considered in the theoretical score,
    # but it is not included in the current reward function.
    age_days = pd.to_numeric(issues_df["age_days"], errors="coerce").fillna(0).to_numpy()
    age_score = np.log1p(age_days)

    # Reward-design constants for the theoretical model.
    alpha = 2.0
    beta = 0.1

    # Earlier experimental version:
    # theory_score = severity_score - alpha * effort_score + beta * age_score

    # Current theoretical reward:
    # prioritize high-severity issues with lower remediation effort.
    theory_score = severity_score - alpha * effort_score

    # Metadata passed to the environment.
    meta = {"theory_score": theory_score, "issue_ids": issues_df["issue_id"].tolist()}

    # --------------------------------------------------
    # 3. Build the RL environment
    # --------------------------------------------------
    # Limit the number of recommendations to Top-5 per snapshot.
    max_recommend = 5   # top-5
    def make_env():
        # Create one TDEnv instance for the current snapshot.
        return TDEnv(X,meta,mode="theory",max_select=max_recommend)

    # Wrap the environment in a vectorized SB3 interface.
    env = make_vec_env(make_env, n_envs=1)

    # --------------------------------------------------
    # 4. Train PPO with the custom logger
    # --------------------------------------------------
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.set_logger(new_logger)

    # Main training budgets used in the thesis:
    # 5,000 / 15,000 / 50,000 timesteps
    model.learn(total_timesteps=50000)

    # --------------------------------------------------
    # 5. Evaluate the learned policy after training
    # --------------------------------------------------
    obs = env.reset()
    actions = []

    # Track already selected indices to avoid duplicates in the final Top-K list.
    picked = set()

    for _ in range(max_recommend):
        action, _ = model.predict(obs, deterministic=True)
        a0 = int(action[0]) if hasattr(action, "__len__") else int(action)

        # If the model repeats an already selected issue,
        # fall back to the first unselected candidate.
        if a0 in picked:
            for cand in range(len(meta["issue_ids"])):
                if cand not in picked:
                    a0 = cand
                    break

        picked.add(a0)
        issue_id = meta["issue_ids"][a0]
        r0 = float(meta["theory_score"][a0])

        # Store the selected issue together with its main attributes.
        actions.append({
            "snapshot": month,
            "issue_id": issue_id,
            "reward": r0,
            "severity": issues_df.loc[issues_df["issue_id"] == issue_id, "severity"].values[0],
            "effort_minutes": issues_df.loc[issues_df["issue_id"] == issue_id, "effort_minutes"].values[0],
            "created_at": issues_df.loc[issues_df["issue_id"] == issue_id, "created_at"].values[0],
        })

        # Advance the environment by executing the selected action.
        obs, _, _, _ = env.step([a0])

    # Sort the selected issues by theoretical reward in descending order.
    actions_sorted = sorted(actions, key=lambda x: x["reward"], reverse=True)
    print(f"Snapshot {month}: Top-{max_recommend} → {actions_sorted}")
    all_results.extend(actions_sorted)

# -------------------------
# 6. Save results
# -------------------------
if all_results:
    result_df = pd.DataFrame(all_results)
    result_df.to_csv("theory.csv", index=False)
    print("Saved theory rankings to theory.csv")

# --------------------------------------------------
# 7. Convert SB3 logger output into a reward-curve file
# --------------------------------------------------
try:
    progress = pd.read_csv(f"{log_dir}/progress.csv")
    if "rollout/ep_rew_mean" in progress.columns:
        # Keep the columns needed for plotting the reward curve.
        df_rewards = progress[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna()
        df_rewards.rename(columns={
            "time/total_timesteps": "timesteps",
            "rollout/ep_rew_mean": "mean_reward"
        }, inplace=True)

        df_rewards.to_csv("reward_curve.csv", index=False)

        # Plot the learning curve used in the thesis results chapter.
        plt.figure(figsize=(12, 10))
        plt.plot(df_rewards["timesteps"], df_rewards["mean_reward"],
                 label="Reward curve",
                 linewidth=0.8,  # 线条更细
                 alpha=0.6,  # 半透明，减少重叠感
                 color="royalblue")  # 蓝色线条可读性好
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Mean Episode Reward", fontsize=12)
        plt.legend(loc="upper left")
        plt.savefig("reward_curve.png", dpi=300)
        print("Saved reward curve to reward_curve.png")
    else:
        print("⚠️ progress.csv did not contain 'rollout/ep_rew_mean'")
except FileNotFoundError:
    print("⚠️ progress.csv not found in logs/")