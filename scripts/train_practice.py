# Abandoned !! Adopt train_practice_groundtruth.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tdppo.datasets import load_github_issues, build_monthly_snapshots
from tdppo.featurizer import featurize, build_dev_rank_map
from tdppo.env import TDEnv

# 1. Load GitHub issues (example repo: aws)
df = load_github_issues("aws/aws-sdk-java-v2", max_issues=500)
snapshots = build_monthly_snapshots(df)

# 2. Iterate snapshots
for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues)")
        continue

    print(f"Training on snapshot {month}, issues={len(issues_df)}")

    # 3. Feature engineering (practice mode: use age_days only)
    X = featurize(issues_df, mode="practice")

    # 4. Build developer rank map (reward signal)
    dev_rank_map = build_dev_rank_map(issues_df)

    meta = {
        "dev_rank": dev_rank_map,
        "issue_ids": issues_df["issue_id"].tolist()
    }

    # 5. Build RL environment
    max_recommend = 10  # 20?
    def make_env():
        return TDEnv(X, meta, mode="practice", max_select=max_recommend)

    env = make_vec_env(make_env, n_envs=1)

    # 6. Train PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)

    model.save(f"models/practice_ppo_{month.strftime('%Y%m')}")

    # 7. Evaluate
    # eval_env = TDEnv(X, meta, mode="practice", max_select=max_recommend)
    obs = env.reset()
    actions = []
    picked = set()

    for step in range(max_recommend):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        a0 = int(action[0]) if hasattr(action, "__len__") else int(action)
        r0 = float(rewards[0]) if hasattr(rewards, "__len__") else float(rewards)

        if a0 in picked:
            continue
        picked.add(a0)
        actions.append((meta["issue_ids"][a0], r0))

        if dones[0] if hasattr(dones, "__len__") else dones:
            break

    actions_sorted = sorted(actions, key=lambda x: x[1], reverse=True)
    print(f"Snapshot {month}: Practice Top-10 → {actions_sorted}")
