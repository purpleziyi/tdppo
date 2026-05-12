# compare_tms_pms_distributions.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0) Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

THEORY_CSV = os.path.join(
    BASE_DIR,
    "TMS outputs",
    "theory tms Top5 4000issues 50000steps.csv"
)

PMS_CSV = os.path.join(
    BASE_DIR,
    "PMS outputs",
    "pms Top5 500issues 50000steps.csv"
)

OUT_DIR = os.path.join(BASE_DIR, "tms_pms_distribution_outputs")

# Deduplication strategy:
# TMS: keep the row with the highest theoretical reward for each issue_id
# PMS: keep the row with the highest practical reward for each issue_id
DEDUP_THEORY_BY_MAX_REWARD = True
DEDUP_PMS_BY_MAX_REWARD = True

# Severity order used for normalized counting and plotting
SEVERITY_ORDER = ["info", "low", "medium", "high", "blocker"]

# Effort bins (in minutes) used for grouped distribution analysis
EFFORT_BINS = [-0.1, 0, 5, 10, 30, 60, np.inf]
EFFORT_LABELS = ["0", "1-5", "6-10", "11-30", "31-60", ">60"]

# Whether to additionally generate the issue-age boxplot
# This plot is optional and was not used in the thesis text.
PLOT_AGE_DISTRIBUTION = True


# =========================================================
# 1) Helper functions
# =========================================================
def ensure_out_dir(path: str):
    """Create the output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and raise an explicit error if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def parse_datetime_cols(df: pd.DataFrame, cols):
    """Parse selected columns as UTC datetimes where available."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def normalize_severity(series: pd.Series) -> pd.Series:
    """
    Normalize severity labels to lowercase categories.
    Also handles possible alternate labels if they appear.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "minor": "low",
            "major": "medium",
            "critical": "high",
        })
    )


def map_severity_weight(series: pd.Series) -> pd.Series:
    """
    Map normalized severity labels to numeric weights.
    These weights follow the theoretical reward design used in TMS.
    """
    mapping = {
        "blocker": 16,
        "high": 8,
        "medium": 4,
        "low": 2,
        "info": 1
    }
    return normalize_severity(series).map(mapping).fillna(1)


def add_effort_bin(df: pd.DataFrame, effort_col: str = "effort_minutes") -> pd.DataFrame:
    """
    Group effort values into coarse bins for distributional comparison.
    """
    out = df.copy()
    out[effort_col] = pd.to_numeric(out[effort_col], errors="coerce")
    out["effort_bin"] = pd.cut(
        out[effort_col],
        bins=EFFORT_BINS,
        labels=EFFORT_LABELS
    )
    return out


def add_age_days(df: pd.DataFrame, created_col: str = "created_at", snapshot_col: str = "snapshot") -> pd.DataFrame:
    """
    Compute issue age at the time of selection:
    age_days_at_selection = snapshot - created_at
    """
    out = df.copy()

    if snapshot_col in out.columns and created_col in out.columns:
        snap = pd.to_datetime(out[snapshot_col], utc=True, errors="coerce")
        created = pd.to_datetime(out[created_col], utc=True, errors="coerce")
        out["age_days_at_selection"] = (snap - created).dt.days
    else:
        out["age_days_at_selection"] = np.nan

    return out


def deduplicate_theory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate TMS outputs at the issue level.

    If enabled, keep the row with the highest theoretical reward
    for each unique issue_id.
    """
    out = df.copy()

    if "issue_id" not in out.columns:
        raise ValueError("TMS csv must contain 'issue_id'.")
    if "reward" not in out.columns:
        raise ValueError("TMS csv must contain 'reward'.")

    out["reward"] = pd.to_numeric(out["reward"], errors="coerce")

    if DEDUP_THEORY_BY_MAX_REWARD:
        out = out.sort_values(["issue_id", "reward"], ascending=[True, False])
        out = out.drop_duplicates(subset=["issue_id"], keep="first")
    else:
        out = out.drop_duplicates(subset=["issue_id"], keep="first")

    return out.reset_index(drop=True)


def deduplicate_pms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate PMS outputs at the issue level.

    If enabled, keep the row with the highest practical reward
    for each unique issue_id.
    """
    out = df.copy()

    if "issue_id" not in out.columns:
        raise ValueError("PMS csv must contain 'issue_id'.")
    if "reward_practice" not in out.columns:
        raise ValueError("PMS csv must contain 'reward_practice'.")

    out["reward_practice"] = pd.to_numeric(out["reward_practice"], errors="coerce")

    if DEDUP_PMS_BY_MAX_REWARD:
        out = out.sort_values(["issue_id", "reward_practice"], ascending=[True, False])
        out = out.drop_duplicates(subset=["issue_id"], keep="first")
    else:
        out = out.drop_duplicates(subset=["issue_id"], keep="first")

    return out.reset_index(drop=True)


def build_severity_summary(tms_df: pd.DataFrame, pms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparative severity summary for deduplicated TMS and PMS outputs.
    Both counts and proportions are computed.
    """
    tms_counts = normalize_severity(tms_df["severity"]).value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    pms_counts = normalize_severity(pms_df["severity"]).value_counts().reindex(SEVERITY_ORDER, fill_value=0)

    summary = pd.DataFrame({
        "severity": SEVERITY_ORDER,
        "tms_count": tms_counts.values,
        "pms_count": pms_counts.values,
    })

    summary["tms_prop"] = summary["tms_count"] / max(summary["tms_count"].sum(), 1)
    summary["pms_prop"] = summary["pms_count"] / max(summary["pms_count"].sum(), 1)
    summary["prop_diff_tms_minus_pms"] = summary["tms_prop"] - summary["pms_prop"]

    return summary


def build_effort_bin_summary(tms_df: pd.DataFrame, pms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparative effort-bin summary for deduplicated TMS and PMS outputs.
    """
    tms_tmp = add_effort_bin(tms_df)
    pms_tmp = add_effort_bin(pms_df)

    tms_counts = tms_tmp["effort_bin"].value_counts().reindex(EFFORT_LABELS, fill_value=0)
    pms_counts = pms_tmp["effort_bin"].value_counts().reindex(EFFORT_LABELS, fill_value=0)

    summary = pd.DataFrame({
        "effort_bin": EFFORT_LABELS,
        "tms_count": tms_counts.values,
        "pms_count": pms_counts.values,
    })

    summary["tms_prop"] = summary["tms_count"] / max(summary["tms_count"].sum(), 1)
    summary["pms_prop"] = summary["pms_count"] / max(summary["pms_count"].sum(), 1)
    summary["prop_diff_tms_minus_pms"] = summary["tms_prop"] - summary["pms_prop"]

    return summary


def build_numeric_summary(tms_df: pd.DataFrame, pms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compact descriptive statistics used in the thesis summary table.
    """
    tms_eff = pd.to_numeric(tms_df["effort_minutes"], errors="coerce")
    pms_eff = pd.to_numeric(pms_df["effort_minutes"], errors="coerce")

    tms_sev = map_severity_weight(tms_df["severity"])
    pms_sev = map_severity_weight(pms_df["severity"])

    tms_age = pd.to_numeric(tms_df.get("age_days_at_selection", np.nan), errors="coerce")
    pms_age = pd.to_numeric(pms_df.get("age_days_at_selection", np.nan), errors="coerce")

    rows = [
        {
            "group": "TMS",
            "n_unique_issues": len(tms_df),
            "mean_effort": tms_eff.mean(),
            "median_effort": tms_eff.median(),
            "q1_effort": tms_eff.quantile(0.25),
            "q3_effort": tms_eff.quantile(0.75),
            "mean_severity_weight": tms_sev.mean(),
            "median_severity_weight": tms_sev.median(),
            "high_or_blocker_prop": normalize_severity(tms_df["severity"]).isin(["high", "blocker"]).mean(),
            "mean_age_days_at_selection": tms_age.mean(),
            "median_age_days_at_selection": tms_age.median(),
        },
        {
            "group": "PMS",
            "n_unique_issues": len(pms_df),
            "mean_effort": pms_eff.mean(),
            "median_effort": pms_eff.median(),
            "q1_effort": pms_eff.quantile(0.25),
            "q3_effort": pms_eff.quantile(0.75),
            "mean_severity_weight": pms_sev.mean(),
            "median_severity_weight": pms_sev.median(),
            "high_or_blocker_prop": normalize_severity(pms_df["severity"]).isin(["high", "blocker"]).mean(),
            "mean_age_days_at_selection": pms_age.mean(),
            "median_age_days_at_selection": pms_age.median(),
        }
    ]

    return pd.DataFrame(rows)


def plot_grouped_bar(summary_df: pd.DataFrame, x_col: str, y1_col: str, y2_col: str,
                     y_label: str, title: str, out_path: str,
                     label1: str = "TMS", label2: str = "PMS"):
    """
    Generate a grouped bar chart for TMS vs PMS distributions.
    """
    x_labels = summary_df[x_col].tolist()
    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, summary_df[y1_col], width, label=label1)
    plt.bar(x + width / 2, summary_df[y2_col], width, label=label2)
    plt.xticks(x, x_labels)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# 2) Main
# =========================================================
def main():
    ensure_out_dir(OUT_DIR)

    # -------------------------
    # Load data
    # -------------------------
    theory_df = load_csv(THEORY_CSV)
    pms_df = load_csv(PMS_CSV)

    # -------------------------
    # Parse timestamps
    # -------------------------
    theory_df = parse_datetime_cols(theory_df, ["snapshot", "created_at"])
    pms_df = parse_datetime_cols(pms_df, ["snapshot", "created_at", "closed_at"])

    # -------------------------
    # Check required columns
    # -------------------------
    theory_required = {"snapshot", "issue_id", "reward", "severity", "effort_minutes", "created_at"}
    pms_required = {"snapshot", "issue_idx", "issue_id", "reward_practice", "created_at", "closed_at", "severity", "effort_minutes"}

    missing_theory = theory_required - set(theory_df.columns)
    missing_pms = pms_required - set(pms_df.columns)

    if missing_theory:
        raise ValueError(f"TMS csv missing required columns: {missing_theory}")
    if missing_pms:
        raise ValueError(f"PMS csv missing required columns: {missing_pms}")

    # -------------------------
    # Normalize / clean
    # -------------------------
    theory_df["severity"] = normalize_severity(theory_df["severity"])
    pms_df["severity"] = normalize_severity(pms_df["severity"])

    theory_df["effort_minutes"] = pd.to_numeric(theory_df["effort_minutes"], errors="coerce")
    pms_df["effort_minutes"] = pd.to_numeric(pms_df["effort_minutes"], errors="coerce")

    theory_df = theory_df.dropna(subset=["issue_id", "severity", "effort_minutes"]).copy()
    pms_df = pms_df.dropna(subset=["issue_id", "severity", "effort_minutes"]).copy()

    # -------------------------
    # Compute age at selection
    # -------------------------
    # This step converts snapshot-level outputs into issue-level outputs
    # by retaining one representative row per issue.
    theory_df = add_age_days(theory_df)
    pms_df = add_age_days(pms_df)

    # Save deduplicated issue-level outputs.
    # These CSVs are useful for exploratory inspection but were NOT used directly
    # in the thesis text.
    theory_unique = deduplicate_theory(theory_df)
    pms_unique = deduplicate_pms(pms_df)

    # Save deduplicated issue-level outputs
    theory_unique.to_csv(os.path.join(OUT_DIR, "theory_unique_issue_outputs.csv"), index=False)
    pms_unique.to_csv(os.path.join(OUT_DIR, "pms_unique_issue_outputs.csv"), index=False)

    # -------------------------
    # Build summaries used in the thesis
    # -------------------------
    severity_summary = build_severity_summary(theory_unique, pms_unique)
    effort_bin_summary = build_effort_bin_summary(theory_unique, pms_unique)
    numeric_summary = build_numeric_summary(theory_unique, pms_unique)

    severity_summary.to_csv(os.path.join(OUT_DIR, "severity_distribution_tms_vs_pms.csv"), index=False)
    effort_bin_summary.to_csv(os.path.join(OUT_DIR, "effort_distribution_tms_vs_pms.csv"), index=False)
    numeric_summary.to_csv(os.path.join(OUT_DIR, "summary_tms_vs_pms.csv"), index=False)

    # -------------------------
    # Main thesis plots
    # -------------------------
    plot_grouped_bar(
        summary_df=severity_summary,
        x_col="severity",
        y1_col="tms_prop",
        y2_col="pms_prop",
        y_label="Proportion",
        title="Severity Distribution: TMS vs PMS Outputs",
        out_path=os.path.join(OUT_DIR, "severity_distribution_tms_vs_pms.png"),
        label1="TMS-selected",
        label2="PMS-selected",
    )

    plot_grouped_bar(
        summary_df=effort_bin_summary,
        x_col="effort_bin",
        y1_col="tms_prop",
        y2_col="pms_prop",
        y_label="Proportion",
        title="Effort Distribution: TMS vs PMS Outputs",
        out_path=os.path.join(OUT_DIR, "effort_distribution_tms_vs_pms.png"),
        label1="TMS-selected",
        label2="PMS-selected",
    )

    # -------------------------
    # Console output
    # -------------------------
    print("\n=== Finished: TMS vs PMS distribution comparison ===")
    print(f"TMS unique issues: {len(theory_unique)}")
    print(f"PMS unique issues: {len(pms_unique)}")
    print(f"Outputs saved to: {OUT_DIR}")

    print("\nGenerated files:")
    generated = [
        "theory_unique_issue_outputs.csv",
        "pms_unique_issue_outputs.csv",
        "severity_distribution_tms_vs_pms.csv",
        "effort_distribution_tms_vs_pms.csv",
        "summary_tms_vs_pms.csv",
        "severity_distribution_tms_vs_pms.png",
        "effort_distribution_tms_vs_pms.png",
        "effort_boxplot_tms_vs_pms.png",
    ]
    if PLOT_AGE_DISTRIBUTION:
        generated.append("age_boxplot_tms_vs_pms.png")

    for fn in generated:
        print(" -", os.path.join(OUT_DIR, fn))


if __name__ == "__main__":
    main()