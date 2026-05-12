import pandas as pd
from tdppo.datasets import load_github_issues, build_monthly_snapshots

# --------------------------------------------------
# 1. Load GitHub issues from the target repository
# --------------------------------------------------
df = load_github_issues("aws/aws-sdk-java-v2", max_issues=2000)

# --------------------------------------------------
# 2. Build monthly backlog snapshots
# --------------------------------------------------
snapshots = build_monthly_snapshots(df)

# Number of issues retained per month in the empirical closure baseline.
top_k = 5   # 10

# Collect Top-K closure results from all monthly snapshots.
all_results = []

# --------------------------------------------------
# 3. Reconstruct monthly empirical closure order
# --------------------------------------------------
for month, issues_df in snapshots.items():
    # Define the time window of the current month.
    month_start = month
    month_end = (month + pd.offsets.MonthBegin(1))

    # Only considers issues that have a non-null "closed_at" timestamp
    closed = issues_df.dropna(subset=["closed_at"]).copy()

    # Convert closed_at to UTC datetime for reliable time filtering.
    closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)

    # Keep only issues closed within the current month.
    closed = closed[(closed["closed_at"] >= month_start) & (closed["closed_at"] < month_end)]

    if closed.empty:
        print(f"Skipping snapshot {month} (no closed issues)")
        continue

    # Sort issues by closure time in ascending order,
    # so earlier closures receive smaller ranks.
    closed = closed.sort_values("closed_at")

    # Assign empirical closure ranks starting from 1.
    closed["rank"] = range(1, len(closed) + 1)

    # Keep only the Top-K earliest closed issues for the current month.
    top_issues = closed.head(top_k)[["issue_id", "created_at", "closed_at", "labels", "rank"]]

    print(f"\nSnapshot {month}: Ground truth Top-{top_k}")
    print(top_issues.to_string(index=False))

    # Add the snapshot identifier before saving the monthly result.
    top_issues = top_issues.copy()
    top_issues["snapshot"] = month  # Save snapshot date
    all_results.append(top_issues)

# --------------------------------------------------
# 4. Save the empirical closure baseline to CSV
# --------------------------------------------------
if all_results:
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv("groundtruth_practice.csv", index=False)
    print("\n Saved ground-truth rankings to groundtruth_practice.csv")


