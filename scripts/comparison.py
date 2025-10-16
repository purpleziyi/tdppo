import pandas as pd
from math import log2

# 读你的结果
practice = pd.read_csv("practice_aws Top5 600issues 50000steps.csv")      # RL practice Top5
gt = pd.read_csv("groundtruth_practice_aws Top10.csv")                    # true monthly Top10

# 统一月份键（假设列名里 snapshot 是可解析时间）
practice["month"] = pd.to_datetime(practice["snapshot"], utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)
gt["month"]       = pd.to_datetime(gt["snapshot"],       utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)

# 每月算 Precision@K 和 NDCG@K（K=5）
def ndcg_at_k(pred_ids, true_ids, k=5):
    dcg = 0.0
    for i, iid in enumerate(pred_ids[:k], start=1):
        rel = 1.0 if iid in true_ids else 0.0
        dcg += rel / log2(i + 1)
    # IDCG: 全 relevant 在前（最多 min(k, |true|) 个）
    m = min(k, len(true_ids))
    idcg = sum(1.0 / log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg if idcg > 0 else 0.0

rows = []
for m, g in gt.groupby("month"):
    true_ids = set(g["issue_id"].astype(str))
    p = practice[practice["month"] == m].sort_values("reward_practice", ascending=False)
    pred_ids = p["issue_id"].astype(str).tolist()

    K = 5
    hit = sum(1 for x in pred_ids[:K] if x in true_ids)
    prec = hit / K
    ndcg = ndcg_at_k(pred_ids, true_ids, k=K)

    rows.append({"month": m, "precision@5": prec, "ndcg@5": ndcg, "hits": hit, "gt_size": len(true_ids)})

eval_df = pd.DataFrame(rows).sort_values("month")
print(eval_df)
print("\nOverall mean:",
      {"precision@5": eval_df["precision@5"].mean(),
       "ndcg@5": eval_df["ndcg@5"].mean()})
