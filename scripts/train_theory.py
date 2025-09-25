import numpy as np
import pandas as pd

from tdppo.datasets import load_sonarcloud_issues, build_monthly_snapshots_sonar
from tdppo.featurizer import featurize, map_severity, normalize
from tdppo.env import TDEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. Load SonarQube issues
df = load_sonarcloud_issues("aws_aws-sdk-java-v2", max_issues=1500)
snapshots = build_monthly_snapshots_sonar(df)

# 2. Iterate all the month in the snapshots
for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues)")
        continue

    print(f"Training on snapshot {month}, issues={len(issues_df)}")

    # 3. Feature engineering
    X = featurize(issues_df, mode="theory")

    severity_score = map_severity(issues_df["severity"])   # Map severity levels to numeric scores
    effort_score = normalize(issues_df["effort_minutes"])   # Normalize effort estimates (lower effort → higher reward)
    # age_days = issues_df["age_days"].fillna(0).to_numpy()   # Age of technical debt in days (older debt → higher interest)
    # age_score = log1p(age_days)    # Use log1p to avoid extreme values (e.g., log(1+age_days))

    # age (days) → use NumPy's vectorized log1p on a numeric ndarray
    age_days = pd.to_numeric(issues_df["age_days"], errors="coerce").fillna(0).to_numpy()
    age_score = np.log1p(age_days)  # vectorized, Use log1p to avoid extreme values (e.g., log(1+age_days))

    alpha = 2.0  # weight for effort penalty
    beta = 0.1   # weight for age bonus

    # Combine into the theoretical reward formula:
    # reward = severity_weight - α * normalized_effort + β * log1p(age_days)
    theory_score = severity_score - alpha * effort_score + beta * age_score
    # theory_score = severity_score - alpha * effort_score   # simple formular

    # Pack into meta dictionary (used by TDEnv)
    meta = {"theory_score": theory_score, "issue_ids": issues_df["issue_id"].tolist()}

    # 4. Build environment
    max_recommend = 5  # 10?
    def make_env():
        return TDEnv(X, meta, mode="theory", max_select=max_recommend)

    env = make_vec_env(make_env, n_envs=1)

    # 5. Train PPO
    # Initialize PPO model with a simple Multi-Layer Perceptron (MLP) policy
    # verbose=1 means it will print training progress
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=15000)

    # model.save(f"models/theory_ppo_{month.strftime('%Y%m')}")

    # 6. Evaluate
    obs = env.reset()
    actions = []
    picked = set()  # 避免重复 issue

    for _ in range(max_recommend):
        action, _ = model.predict(obs, deterministic=True)
        a0 = int(action[0]) if hasattr(action, "__len__") else int(action)

        if a0 in picked:
            # fallback
            for cand in range(len(meta["issue_ids"])):
                if cand not in picked:
                    a0 = cand
                    break

        picked.add(a0)
        r0 = float(meta["theory_score"][a0])
        actions.append((meta["issue_ids"][a0], r0))

        # 不 break，即使环境 done 也继续
        obs, _, _, _ = env.step([a0])  # 保持 obs 更新，但忽略 reward/done

    actions_sorted = sorted(actions, key=lambda x: x[1], reverse=True)
    print(f"Snapshot {month}: Top-{max_recommend} → {actions_sorted}")