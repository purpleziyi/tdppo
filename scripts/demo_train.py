import requests
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tdppo.datasets import build_monthly_snapshots, build_monthly_snapshots_sonar
from tdppo.featurizer import featurize, map_severity, normalize, build_dev_rank_map
from tdppo.env import TDEnv

# =========================================================
# 1. Fetch GitHub issues
# =========================================================
# We use GitHub's REST API to request issues from a repository.
# Example repository: "scikit-learn/scikit-learn"
# repo = "scikit-learn/scikit-learn"   # github - practice
# url = f"https://api.github.com/repos/{repo}/issues"  # github - practice
url = "https://sonarcloud.io/api/issues/search"    # SonarQube - theory
page = 1
rows = []
max_issues = 800

while True:
    # params = {"state":"all", "per_page":50, "page": page} # Get both open and closed issues, limit to 50
    params = { "componentKeys": "microsoft_kiota", "statuses": "OPEN", "ps": 100, "p": 1 }
    r = requests.get(url, params=params)
    data = r.json()  # Convert JSON response to Python list of dicts
    issues = data.get("issues", [])

    if not isinstance(issues, list):
        print("GitHub API error:", issues)  # 打印出来调试
        break

    # stop if no more issues
    if not issues or len(rows) >= max_issues:
        break

    # Convert issues into structured rows for a DataFrame

    for issue in issues:
        if "pull_request" in issue:  # skip PRs
            continue

        # rows.append({     # for Github-practice
        #     "issue_id": issue["number"],                       # Unique issue number in GitHub repo
        #     "title": issue["title"],                           # Short summary
        #     "labels": ";".join([l["name"] for l in issue["labels"]]),  # Concatenate all labels
        #     "state": issue["state"],                           # "open" or "closed"
        #     "created_at": issue["created_at"],                 # Creation timestamp
        #     "closed_at": issue.get("closed_at"),               # Closing timestamp (may be None)
        #     # Artificial fields (manually created for toy example)
        #     "severity": np.random.choice(["blocker", "high", "medium", "low", "info"]),  # Simulated severity
        #     "effort_minutes": np.random.randint(1, 300),      # Simulated effort estimation
        #     "text": (issue.get("title") or "") + " " + (issue.get("body") or ""),  # Title + body as text
        #     "age_days": (
        #         (pd.to_datetime(issue.get("closed_at") or pd.Timestamp.utcnow(), utc=True)
        #          - pd.to_datetime(issue["created_at"], utc=True)).days
        #     ) # calculated with GitHub timestamp
        # })

        # for SonarQubeCloud - theory
        effort = issue.get("effort", "0min")
        minutes = 0
        if "h" in effort:
            parts = effort.split("h")
            h = int(parts[0])
            minutes += h * 60
            if "min" in parts[1]:
                m = int(parts[1].replace("min", "").strip() or 0)
                minutes += m
        elif "min" in effort:
            minutes = int(effort.replace("min", "").strip() or 0)

        # Try to use Quality Impact Severity if available
        impact_severity = None
        impacts = issue.get("impacts", [])
        if impacts and isinstance(impacts, list):
            impact_severity = impacts[0].get("severity")

        rows.append({
            "issue_id": issue.get("key"),
            "title": issue.get("message"),
            "labels": ";".join(issue.get("tags", [])),
            "state": "OPEN".lower(),
            "created_at": issue.get("creationDate"),
            "closed_at": None,  # SonarCloud "OPEN" issues usually no closed date
            "severity": (impact_severity or issue.get("severity", "info")),
            "effort_minutes": minutes,
            "text": issue.get("message", ""),
            "age_days": (
                (pd.to_datetime(pd.Timestamp.utcnow(), utc=True) - pd.to_datetime(issue.get("creationDate"),
                                                                                  utc=True)).days
            )
        })
    page += 1

# Create a Pandas DataFrame from collected rows
df = pd.DataFrame(rows)
# snapshots = build_monthly_snapshots(df)  # for Github - practice
snapshots = build_monthly_snapshots_sonar(df)   # for SonarQube - theory

for month, issues_df in snapshots.items():
    if issues_df.empty:
        print(f"Skipping snapshot {month} (no issues)")
        continue

    print(f"Training on snapshot {month}, issues={len(issues_df)}")
    # =========================================================
    # 2. Construct features (X) and metadata (meta)
    # =========================================================
    # Featurize issues into a numeric matrix
    X = featurize(issues_df, mode="theory")
    # X = featurize(issues_df, mode="practice")

    # Define a "theory score" as severity - normalized effort
    theory_score = map_severity(issues_df["severity"]) - 2 * normalize(issues_df["effort_minutes"])
    # dev_rank_map = build_dev_rank_map(issues_df)

    # Metadata dictionary used by the environment
    meta = {
        "theory_score": theory_score,
        "issue_ids": issues_df["issue_id"].tolist()
    }
    # meta = {
    #         "dev_rank": dev_rank_map,
    #         "issue_ids": issues_df["issue_id"].tolist()
    #     }

    # =========================================================
    # 3. Wrap the environment
    # =========================================================
    # Define a function that creates an instance of the custom environment
    max_recommend = 5
    def make_env():
        """
        Create a TDEnv environment.

        Returns:
            TDEnv: Custom environment instance with fixed features and metadata.
        """
        return TDEnv(X, meta, mode="theory", max_select = max_recommend )
        # return TDEnv(X, meta, mode="practice", max_select=5)

    # Wrap environment with Stable-Baselines3 helper (supports parallelization)
    env = make_vec_env(make_env, n_envs=1)

    # =========================================================
    # 4. Train PPO agent
    # =========================================================
    # Initialize PPO model with a simple Multi-Layer Perceptron (MLP) policy
    # verbose=1 means it will print training progress
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model for certain timesteps
    model.learn(total_timesteps=10000)

    # =========================================================
    # 5. Inference (using the trained model)
    # =========================================================
    # Reset environment to get initial observation
    obs = env.reset()

    actions = []
    picked = set()

    # Run for at most 5 steps
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