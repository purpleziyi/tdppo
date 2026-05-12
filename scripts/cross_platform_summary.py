import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 可配置路径 =========
THEORY_PATH = "TMS outputs/theory tms Top5 4000issues 50000steps.csv"                       # e.g. "theory_aws 1500issues 50000steps.csv"
GITHUB_GT_PATH = "groundtruth_practice_aws Top5 Github.csv"      # e.g. "groundtruth_practice_aws Top10.csv"

OUT_MD = "cross_platform_summary.md"
OUT_FIG = "cross_platform_summary.png"

# ========= 工具函数 =========

def ensure_cols(df, need, path):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺少列: {miss}")

def normalize_severity(s):
    """把各种口径的严重度归一到 {BLOCKER, HIGH, MEDIUM, LOW, INFO, OTHER}."""
    if not isinstance(s, str):
        return "OTHER"
    t = s.strip().upper()
    # 常见同义
    if t in {"BLOCKER"}:
        return "BLOCKER"
    if t in {"CRITICAL", "HIGH"}:
        return "HIGH"
    if t in {"MAJOR", "MEDIUM"}:
        return "MEDIUM"
    if t in {"MINOR", "LOW"}:
        return "LOW"
    if t in {"INFO"}:
        return "INFO"
    return "OTHER"

def extract_p_level(labels):
    """从 GitHub labels 字符串里抽取 p0~p3；无则返回 None."""
    if not isinstance(labels, str):
        return None
    m = re.search(r"\bp([0-3])\b", labels.lower())
    return m.group(0) if m else None

def to_table(df_counts, total, name_col="class"):
    """给定计数Series，返回带百分比的DataFrame（降序）"""
    df = df_counts.rename("count").reset_index().rename(columns={"index": name_col})
    df["percent"] = (df["count"] / total * 100.0).round(2)
    return df.sort_values("count", ascending=False).reset_index(drop=True)

# ========= 1) 读取数据 =========

theory = pd.read_csv(THEORY_PATH)
ensure_cols(theory, ["severity"], THEORY_PATH)

gt = pd.read_csv(GITHUB_GT_PATH)
ensure_cols(gt, ["labels"], GITHUB_GT_PATH)

# ========= 2) 统计 Sonar 各 severity =========

sev_norm = theory["severity"].map(normalize_severity)
sev_counts = sev_norm.value_counts()
sev_total = int(sev_counts.sum())

# 保证固定顺序
sev_order = ["BLOCKER", "HIGH", "MEDIUM", "LOW", "INFO", "OTHER"]
for k in sev_order:
    if k not in sev_counts:
        sev_counts[k] = 0
sev_counts = sev_counts[sev_order]

sev_table = to_table(sev_counts, sev_total, name_col="severity")

# ========= 3) 统计 GitHub p-level =========

p_levels = gt["labels"].map(extract_p_level)
# 统计 p0~p3
p_counts = p_levels.value_counts(dropna=False)
# 把 NaN 归一成 no_p_label
no_p = int(p_counts.get(np.nan, 0))
p0 = int(p_counts.get("p0", 0))
p1 = int(p_counts.get("p1", 0))
p2 = int(p_counts.get("p2", 0))
p3 = int(p_counts.get("p3", 0))
p_order = ["p0", "p1", "p2", "p3", "no_p_label"]
p_series = pd.Series({"p0": p0, "p1": p1, "p2": p2, "p3": p3, "no_p_label": no_p})
p_total = int(p_series.sum())
p_table = to_table(p_series, p_total, name_col="p_level")

# ========= 4) 导出 Markdown 表格 =========

lines = []
lines.append("# Cross-Platform Summary (AWS)\n")
lines.append(f"- Theory (Sonar) file: `{os.path.basename(THEORY_PATH)}`  \n- Practice (GitHub) file: `{os.path.basename(GITHUB_GT_PATH)}`\n")

lines.append("## SonarQube Severity Distribution\n")
lines.append(f"**Total:** {sev_total}\n")
lines.append("| severity | count | percent |")
lines.append("|---|---:|---:|")
for _, r in sev_table.iterrows():
    lines.append(f"| {r['severity']} | {int(r['count'])} | {r['percent']}% |")

lines.append("\n## GitHub Priority Label Distribution (p-level)\n")
lines.append(f"**Total:** {p_total}  （`no_p_label` 表示该 issue 未标注 p0~p3）\n")
lines.append("| p_level | count | percent |")
lines.append("|---|---:|---:|")
for _, r in p_table.iterrows():
    lines.append(f"| {r['p_level']} | {int(r['count'])} | {r['percent']}% |")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved markdown: {OUT_MD}")

# ========= 5) 论文可插图（双子图柱状图） =========

plt.figure(figsize=(11,5))

# 左图：Sonar severities
plt.subplot(1,2,1)
x1 = np.arange(len(sev_order))
vals1 = [int(sev_counts[k]) for k in sev_order]
plt.bar(x1, vals1)
plt.xticks(x1, sev_order, rotation=25, ha="right")
plt.ylabel("Count")
plt.title("SonarQube: Severity Distribution")

# 在柱顶标注百分比
for xi, v in zip(x1, vals1):
    if sev_total > 0:
        pct = 100.0 * v / sev_total
        plt.text(xi, v, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

# 右图：GitHub p-levels
plt.subplot(1,2,2)
x2 = np.arange(len(p_order))
vals2 = [int(p_series[k]) for k in p_order]
plt.bar(x2, vals2)
plt.xticks(x2, p_order, rotation=25, ha="right")
plt.ylabel("Count")
plt.title("GitHub: Priority Labels (p0–p3 / no_p_label)")

for xi, v in zip(x2, vals2):
    if p_total > 0:
        pct = 100.0 * v / p_total
        plt.text(xi, v, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Saved figure: {OUT_FIG}")

# ========= 6) 终端友好输出（可选） =========
print("\n--- SonarQube Severity ---")
print(sev_table.to_string(index=False))
print("\n--- GitHub p-level ---")
print(p_table.to_string(index=False))
