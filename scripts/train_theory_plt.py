import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tdppo.datasets import load_sonarcloud_issues, build_monthly_snapshots_sonar
from tdppo.featurizer import featurize, map_severity, normalize
from tdppo.env import TDEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# -------------------------
# 1. Load SonarQube issues
# -------------------------
df = load_sonarcloud_issues("aws_aws-sdk-java-v2", max_issues=1500)
snapshots = build_monthly_snapshots_sonar(df)

all_results = []

# directory for SB3 logs
log_dir = "logs/"  # put logger into scripts/train_theory/logs
new_logger = configure(log_dir, ["stdout", "csv"])   # 输出到终端和 CSV(progress.csv)

for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues)")
        continue

    print(f"Training on snapshot {month}, issues={len(issues_df)}")

    # -------------------------
    # 2. Feature engineering
    # -------------------------
    X = featurize(issues_df, mode="theory")

    severity_score = map_severity(issues_df["severity"])
    effort_score = normalize(issues_df["effort_minutes"])
    age_days = pd.to_numeric(issues_df["age_days"], errors="coerce").fillna(0).to_numpy()
    age_score = np.log1p(age_days)

    alpha = 2.0
    beta = 0.1

    theory_score = severity_score - alpha * effort_score + beta * age_score

    meta = {"theory_score": theory_score, "issue_ids": issues_df["issue_id"].tolist()}

    # -------------------------
    # 3. Build environment
    # -------------------------
    max_recommend = 5
    def make_env():
        return TDEnv(X,meta,mode="theory",max_select=max_recommend)

    env = make_vec_env(make_env, n_envs=1)

    # -------------------------
    # 4. Train PPO with custom logger
    # -------------------------
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.set_logger(new_logger)
    model.learn(total_timesteps=15000)  # 5000 , 15000 , 50000

    # -------------------------
    # 5. Evaluate after training
    # -------------------------
    obs = env.reset()
    actions = []
    picked = set()   # 去重复

    for _ in range(max_recommend):
        action, _ = model.predict(obs, deterministic=True)
        a0 = int(action[0]) if hasattr(action, "__len__") else int(action)

        if a0 in picked:
            for cand in range(len(meta["issue_ids"])):
                if cand not in picked:
                    a0 = cand
                    break

        picked.add(a0)
        issue_id = meta["issue_ids"][a0]
        r0 = float(meta["theory_score"][a0])
        actions.append({
            "snapshot": month,
            "issue_id": issue_id,
            "reward": r0,
            "severity": issues_df.loc[issues_df["issue_id"] == issue_id, "severity"].values[0],
            "effort_minutes": issues_df.loc[issues_df["issue_id"] == issue_id, "effort_minutes"].values[0],
            "created_at": issues_df.loc[issues_df["issue_id"] == issue_id, "created_at"].values[0],
        })

        obs, _, _, _ = env.step([a0])

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

# -------------------------
# 7. Process logger CSV → reward_curve
# -------------------------
try:
    progress = pd.read_csv(f"{log_dir}/progress.csv")
    if "rollout/ep_rew_mean" in progress.columns:
        df_rewards = progress[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna()
        df_rewards.rename(columns={
            "time/total_timesteps": "timesteps",
            "rollout/ep_rew_mean": "mean_reward"
        }, inplace=True)

        df_rewards.to_csv("reward_curve.csv", index=False)

        plt.figure()
        plt.plot(df_rewards["timesteps"], df_rewards["mean_reward"], label="Reward curve")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.legend()
        plt.savefig("reward_curve.png")
        print("Saved reward curve to reward_curve.png")
    else:
        print("⚠️ progress.csv did not contain 'rollout/ep_rew_mean'")
except FileNotFoundError:
    print("⚠️ progress.csv not found in logs/")