import pandas as pd
from tdppo.datasets import load_github_issues, build_monthly_snapshots

# 1. Load GitHub issues (example repo: aws)
df = load_github_issues("aws/aws-sdk-java-v2", max_issues=600)

# 2. Build monthly snapshots
snapshots = build_monthly_snapshots(df)

top_k = 10
all_results = []  # for saving Top-K issues of each month

# 3. Iterate over snapshots
for month, issues_df in snapshots.items():
    # 确定当月的时间范围
    month_start = month
    month_end = (month + pd.offsets.MonthBegin(1))

    # Only considers issues that have a non-null "closed_at" timestamp
    closed = issues_df.dropna(subset=["closed_at"]).copy()
    closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)  # Convert closed_at to datetime
    closed = closed[(closed["closed_at"] >= month_start) & (closed["closed_at"] < month_end)]  # 只保留当月关闭的 issues

    if closed.empty:
        print(f"Skipping snapshot {month} (no closed issues)")
        continue

    # Sorts these issues by closure time in ascending order
    closed = closed.sort_values("closed_at")
    # Assigns ranks incrementally
    closed["rank"] = range(1, len(closed) + 1)

    # Pick the first K issues
    top_issues = closed.head(top_k)[["issue_id", "created_at", "closed_at", "rank"]]

    print(f"\nSnapshot {month}: Ground truth Top-{top_k}")
    print(top_issues.to_string(index=False))

    # Adds to global result: all_results
    top_issues = top_issues.copy()
    top_issues["snapshot"] = month  # Save snapshot date
    all_results.append(top_issues)

# 4. Save all_results to CSV
if all_results:
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv("groundtruth_practice.csv", index=False)
    print("\n Saved ground-truth rankings to groundtruth_practice.csv")


