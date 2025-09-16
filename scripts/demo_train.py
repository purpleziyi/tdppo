import requests
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tdppo.featurizer import featurize, map_severity, normalize, build_dev_rank_map
from tdppo.env import TDEnv

# =========================================================
# 1. Fetch GitHub issues
# =========================================================
# We use GitHub's REST API to request issues from a repository.
# Example repository: "scikit-learn/scikit-learn"
repo = "scikit-learn/scikit-learn"
url = f"https://api.github.com/repos/{repo}/issues"
params = {"state":"all", "per_page":20} # Get both open and closed issues, limit to 20
r = requests.get(url, params=params)
data = r.json()  # Convert JSON response to Python list of dicts

# Convert issues into structured rows for a DataFrame
rows = []
for issue in data:
    rows.append({
        "issue_id": issue["number"],                       # Unique issue number in GitHub repo
        "title": issue["title"],                           # Short summary
        "labels": ";".join([l["name"] for l in issue["labels"]]),  # Concatenate all labels
        "state": issue["state"],                           # "open" or "closed"
        "created_at": issue["created_at"],                 # Creation timestamp
        "closed_at": issue.get("closed_at"),               # Closing timestamp (may be None)
        # Artificial fields (manually created for toy example)
        "severity": np.random.choice(["blocker", "high", "medium", "low", "info"]),  # Simulated severity
        "effort_minutes": np.random.randint(1, 300),      # Simulated effort estimation
        "text": (issue.get("title") or "") + " " + (issue.get("body") or ""),  # Title + body as text
        "age_days": (
            (pd.to_datetime(issue.get("closed_at") or pd.Timestamp.utcnow(), utc=True)
             - pd.to_datetime(issue["created_at"], utc=True)).days
        ) # calculated with GitHub timestamp
    })

# Create a Pandas DataFrame from collected rows
issues_df = pd.DataFrame(rows)

# =========================================================
# 2. Construct features (X) and metadata (meta)
# =========================================================
# Featurize issues into a numeric matrix
X = featurize(issues_df, mode="theory")

# Define a "theory score" as severity - normalized effort
theory_score = map_severity(issues_df["severity"]) - normalize(issues_df["effort_minutes"])

# Metadata dictionary used by the environment
meta = {
    "theory_score": theory_score,
    "issue_ids": issues_df["issue_id"].tolist()
}

# =========================================================
# 3. Wrap the environment
# =========================================================
# Define a function that creates an instance of the custom environment
def make_env():
    """
    Create a TDEnv environment.

    Returns:
        TDEnv: Custom environment instance with fixed features and metadata.
    """
    return TDEnv(X, meta, mode="theory", max_select=5)

# Wrap environment with Stable-Baselines3 helper (supports parallelization)
env = make_vec_env(make_env, n_envs=1)

# =========================================================
# 4. Train PPO agent
# =========================================================
# Initialize PPO model with a simple Multi-Layer Perceptron (MLP) policy
# verbose=1 means it will print training progress
model = PPO("MlpPolicy", env, verbose=1)

# Train the model for certain timesteps
model.learn(total_timesteps=50000)

# =========================================================
# 5. Inference (using the trained model)
# =========================================================
# Reset environment to get initial observation
obs = env.reset()

# Run for at most 5 steps
for _ in range(5):
    # Select action based on current policy
    action, _ = model.predict(obs, deterministic=True)

    # Step through the environment
    obs, reward, done, info = env.step(action)

    print(f"Action={action}, Reward={reward}")

    if done:
        break