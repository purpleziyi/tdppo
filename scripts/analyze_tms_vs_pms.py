import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
TMS_PATH = "TMS outputs/theory tms Top5 4000issues 50000steps.csv"   # train_theory_plt.py output
PMS_PATH = "PMS outputs/pms Top5 500issues 50000steps.csv"   # train_pms_plt.py output
TOP_K = 5
OUT_DIR = Path("analysis_tms_vs_pms")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
tms = pd.read_csv(TMS_PATH, parse_dates=["snapshot", "created_at"])
pms = pd.read_csv(PMS_PATH, parse_dates=["snapshot", "created_at", "closed_at"])

# -----------------------------
# Focus only on PMS months with practice signal
# -----------------------------
pms_signal = pms[pms["reward_practice"] > 0].copy()  # exclude -0.01 reward

signal_months = sorted(pms_signal["snapshot"].unique())   # 按月份去重
print(f"Analyzing PMS practice months: {signal_months}")

# -----------------------------
# Keep Top-K per month
# -----------------------------
tms_topk = (
    tms.sort_values(["snapshot", "reward"], ascending=[True, False])
       .groupby("snapshot")
       .head(TOP_K)
)

pms_topk = (
    pms_signal.sort_values(["snapshot", "reward_practice"], ascending=[True, False])
              .groupby("snapshot")
              .head(TOP_K)
)

# -----------------------------
# Merge for comparison
# -----------------------------
# 标记数据来源
tms_topk["TMS_selected"] = True
pms_topk["PMS_selected"] = True

# -----------------------------
# Rename columns to wide format
# -----------------------------
# 定义对比时需要保留的共同列
tms_wide = tms_topk.rename(columns={
    "severity": "TMS_severity",
    "effort_minutes": "TMS_effort",
    "created_at": "TMS_created"
})[
    ["snapshot", "issue_id",
     "TMS_selected", "TMS_severity", "TMS_effort", "TMS_created"]
]

pms_wide = pms_topk.rename(columns={
    "severity": "PMS_severity",
    "effort_minutes": "PMS_effort",
    "created_at": "PMS_created"
})[
    ["snapshot", "issue_id",
     "PMS_selected", "PMS_severity", "PMS_effort", "PMS_created"]
]

# -----------------------------
# Merge (outer join = union of issues)
# -----------------------------
comparison_wide = pd.merge(
    tms_wide,
    pms_wide,
    on=["snapshot", "issue_id"],
    how="outer"
)

# -----------------------------
# Fill missing selections
# -----------------------------
comparison_wide["TMS_selected"] = comparison_wide["TMS_selected"].fillna(False)
comparison_wide["PMS_selected"] = comparison_wide["PMS_selected"].fillna(False)

# -----------------------------
# Sort for readability
# -----------------------------
comparison_wide = comparison_wide.sort_values(
    ["snapshot", "TMS_selected", "PMS_selected"],
    ascending=[True, False, False]
)

# -----------------------------
# Save
# -----------------------------
comparison_wide.to_csv(OUT_DIR / "tms_pms_comparison.csv", index=False)

print("Saved wide-format TMS vs PMS comparison to:",
      OUT_DIR / "tms_pms_wide_comparison.csv")



# -----------------------------
# Terminal summary
# -----------------------------
print("\n=== Top-K Overlap per Month ===")
for m in signal_months:
    tms_ids = set(tms_topk[tms_topk["snapshot"] == m]["issue_id"])
    pms_ids = set(pms_topk[pms_topk["snapshot"] == m]["issue_id"])
    overlap = tms_ids & pms_ids
    print(f"{m.date()} | overlap {len(overlap)} / {TOP_K} → {overlap}")

# -----------------------------
# Visualization (Wide-format)
# -----------------------------
def plot_effort_distribution():
    plt.figure(figsize=(6,4))

    tms_effort = comparison_wide.loc[
        comparison_wide["TMS_selected"] == True,
        "TMS_effort"
    ].dropna()

    pms_effort = comparison_wide.loc[
        comparison_wide["PMS_selected"] == True,
        "PMS_effort"
    ].dropna()

    plt.hist(
        tms_effort,
        bins=10,
        alpha=0.6,
        label="TMS (theoretical)"
    )
    plt.hist(
        pms_effort,
        bins=10,
        alpha=0.6,
        label="PMS (practical)"
    )

    plt.xlabel("Effort (minutes)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "effort_distribution_tms_vs_pms.png")
    plt.close()

def plot_severity_distribution():
    plt.figure(figsize=(6,4))

    tms_sev = (
        comparison_wide.loc[comparison_wide["TMS_selected"] == True, "TMS_severity"]
        .value_counts()
    )
    pms_sev = (
        comparison_wide.loc[comparison_wide["PMS_selected"] == True, "PMS_severity"]
        .value_counts()
    )

    df_sev = (
        pd.DataFrame({"TMS": tms_sev, "PMS": pms_sev})
        .fillna(0)
        .sort_index()
    )

    df_sev.plot(kind="bar", alpha=0.75)
    plt.ylabel("Count")
    plt.title("Severity distribution of selected issues")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "severity_distribution_tms_vs_pms.png")
    plt.close()

def plot_divergence_effort():
    plt.figure(figsize=(6,4))

    only_tms = comparison_wide[
        (comparison_wide["TMS_selected"] == True) &
        (comparison_wide["PMS_selected"] == False)
    ]["TMS_effort"].dropna()

    only_pms = comparison_wide[
        (comparison_wide["PMS_selected"] == True) &
        (comparison_wide["TMS_selected"] == False)
    ]["PMS_effort"].dropna()

    plt.hist(only_tms, bins=10, alpha=0.6, label="Only TMS")
    plt.hist(only_pms, bins=10, alpha=0.6, label="Only PMS")

    plt.xlabel("Effort (minutes)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "effort_divergence.png")
    plt.close()

plot_effort_distribution()
plot_severity_distribution()
plot_divergence_effort()

print(f"\nSaved analysis outputs to {OUT_DIR.resolve()}")
