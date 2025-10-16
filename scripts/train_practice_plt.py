import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tdppo.datasets import load_github_issues, build_monthly_snapshots
from tdppo.featurizer import featurize
from tdppo.env import TDEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# -------------------------
# 0) Config
# -------------------------
REPO = "aws/aws-sdk-java-v2"
MAX_ISSUES = 600
TOTAL_TIMESTEPS = 5000          # 可改为 5000 / 15000 / 50000 做对比
MAX_RECOMMEND = 5                # 每月输出前K条推荐
BASE_LOG_DIR = "logs_practice"   # 每月单独子目录，避免 progress.csv 被覆盖

os.makedirs(BASE_LOG_DIR, exist_ok=True)

# -------------------------
# 1) Load GitHub issues & monthly snapshots
# -------------------------
df = load_github_issues(REPO, max_issues=MAX_ISSUES)   # GitHub 实践数据源
snapshots = build_monthly_snapshots(df)                 # 以每月1日的 backlog 作为快照集合

all_rows = []                 # 合并每月 Top-K 结果 → practice.csv
agg_reward_frames = []        # 汇总所有月份的 progress.csv（仅提取关心列）

for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues in backlog)")
        continue

    issues_df = issues_df.reset_index(drop=True)  # 月循环开始后，拿到 issues_df 立刻重置索引

    print(f"\n[Practice-RL] Training on snapshot {month}, backlog size={len(issues_df)}")

    # -------------------------
    # 2) Build practice reward (month-specific rank)
    #    当月关闭的 issues：rank=1..N → reward=1/rank
    #    非当月关闭：reward 使用环境中的小负值（-0.01）
    # -------------------------
    month_start = month
    month_end = month + pd.offsets.MonthBegin(1)

    closed = issues_df.dropna(subset=["closed_at"]).copy()
    closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)
    closed_in_month = closed[(closed["closed_at"] >= month_start) & (closed["closed_at"] < month_end)]
    closed_in_month = closed_in_month.sort_values("closed_at")

    # 将“当月关闭”的行索引映射到 1..N 的 rank（用于奖励）（注意：此时 issues_df 已 reset_index）
    dev_rank_map = {}
    for r, idx in enumerate(closed_in_month.index, start=1):
        dev_rank_map[idx] = r

    # -------------------------
    # 3) Feature engineering (practice mode → TF-IDF + age)
    # -------------------------
    X = featurize(issues_df, mode="practice")  # practice 模式：TF-IDF + age_days 归一化

    meta = {
        "dev_rank": dev_rank_map,  # 环境里将用 1.0/rank
        "issue_ids": issues_df["issue_id"].tolist(),  # 便于回填结果
    }


    # -------------------------
    # 4) Make env (practice mode), set per-month log dir
    # -------------------------
    def make_env():
        return TDEnv(X, meta, mode="practice", max_select=MAX_RECOMMEND)


    env = make_vec_env(make_env, n_envs=1)

    # 每月单独日志目录
    month_tag = month.strftime("%Y-%m")
    month_log_dir = os.path.join(BASE_LOG_DIR, month_tag)
    os.makedirs(month_log_dir, exist_ok=True)
    month_logger = configure(month_log_dir, ["stdout", "csv"])  # 生成 {month}/progress.csv

    # -------------------------
    # 5) Train PPO (practice)
    # -------------------------
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")  # 或 "cpu"
    model.set_logger(month_logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # -------------------------
    # 6) Evaluate greedy Top-K after training
    # -------------------------
    obs = env.reset()
    actions_buffer = []
    picked = set()  # 防止重复挑选

    for _ in range(MAX_RECOMMEND):
        action, _ = model.predict(obs, deterministic=True)
        a0 = int(action[0]) if hasattr(action, "__len__") else int(action)

        if a0 in picked:
            # 兜底：挑选第一个未选过的索引
            for cand in range(len(meta["issue_ids"])):
                if cand not in picked:
                    a0 = cand
                    break

        picked.add(a0)

        # 计算该动作的实践奖励（与环境一致：1/rank 或小负值）
        # 这里仅用于导出结果的参考值
        rank = dev_rank_map.get(a0, None)              # 与 reset 后的索引对齐， 不用rank = dev_rank_map.get(a0, None)
        reward_val = (1.0 / rank) if rank is not None else -0.01

        row = issues_df.iloc[a0]  # ← 关键改动：用 iloc
        issue_id = meta["issue_ids"][a0]
        actions_buffer.append({
            "snapshot": month,
            "issue_idx": a0,
            "issue_id": issue_id,
            "reward_practice": reward_val,
            "created_at": row["created_at"],
            "closed_at": row["closed_at"],
            "labels": row["labels"],
        })

        # 推动环境前进
        obs, _, _, _ = env.step([a0])

    # 将该月 Top-K 以“奖励从高到低”排序并收集
    actions_sorted = sorted(actions_buffer, key=lambda x: x["reward_practice"], reverse=True)
    print(f"[Practice-RL] Snapshot {month}: Top-{MAX_RECOMMEND} →",
          [(r['issue_id'], round(r['reward_practice'], 4)) for r in actions_sorted])
    all_rows.extend(actions_sorted)

    # -------------------------
    # 7) 读取该月 progress.csv，提取关键列并打上月份标签，便于汇总画图
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
# 8) Save practice.csv (Top-K across months)
# -------------------------
if all_rows:
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv("practice.csv", index=False)
    print("\nSaved practice rankings to practice.csv")
else:
    print("\nNo practice results to save (all snapshots were empty or skipped).")

# -------------------------
# 9) Aggregate reward curves across months
# -------------------------
if agg_reward_frames:
    curve_df = pd.concat(agg_reward_frames, ignore_index=True)
    curve_df.to_csv("reward_curve_practice.csv", index=False)

    # 画多条曲线（按月份区分）
    plt.figure(figsize=(8,5))
    for m in curve_df["month"].unique():
        sub = curve_df[curve_df["month"] == m]
        plt.plot(sub["timesteps"], sub["mean_reward"], label=m)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")

    # 图例放在右侧外部，不遮挡线条
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
    plt.tight_layout()  # 自动调整边距，防止被裁剪
    plt.savefig("reward_curve_practice.png")
    print("Saved practice reward curve to reward_curve_practice.png")
else:
    print("No reward curves to aggregate (no progress.csv found).")





