import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def map_severity(severity_series: pd.Series) -> np.ndarray:
    """
    Map categorical severity labels to numeric values.

    Args:
        severity_series (pd.Series): Series containing severity strings
            (e.g., "blocker", "high", "medium", "low", "info").

    Returns:
        np.ndarray: Numeric array corresponding to severity values.
    """
    mapping = {"blocker": 11, "high": 7, "medium": 4, "low": 2, "info": 1}
    return severity_series.str.lower().map(mapping).fillna(1).to_numpy()

def normalize(values:pd.Series) -> np.ndarray:
    """
    Normalize numeric values into the range [0, 1] using Min-Max scaling.

    Args:
        values (pd.Series): Numeric column (e.g., effort, age).

    Returns:
        np.ndarray: Normalized values as a 1D numpy array.
    """
    arr =values.fillna(0).to_numpy().reshape(-1, 1)
    if arr.max() == arr.min():
        # Avoid division by zero if all values are the same
        return np.zeros_like(arr).reshape(-1)
    return MinMaxScaler().fit_transform(arr).reshape(-1)

def featurize(issues_df: pd.DataFrame, mode="theory") -> np.ndarray:
    """
    Extract features from issue data for RL environment.

    Features:
        - Textual content is transformed into TF-IDF vectors (max 20 features).
        - If mode == "theory":
            * Severity is mapped to numeric values.
            * Effort (in minutes) is normalized.
        - If mode != "theory":
            * Issue age (in days) is normalized.

    Args:
        issues_df (pd.DataFrame): DataFrame containing issue attributes.
            Required columns:
                - "text": textual description of the issue.
                - "severity" (for theory mode).
                - "effort_minutes" (for theory mode).
                - "age_days" (for non-theory mode).
        mode (str): Determines which additional features to include.

    Returns:
        np.ndarray: Combined feature matrix (K x d).
    """
    features = []

    # Text feature extraction using TF-IDF
    texts = issues_df["text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=20)
    tfidf = vectorizer.fit_transform(texts).toarray()
    features.append(tfidf)

    if mode == "theory":
        # Map severity levels to numeric values
        sev = map_severity(issues_df["severity"])
        # Normalize effort estimates
        eff = normalize(issues_df["effort_minutes"])
        # Normalize age (days since creation)
        age = normalize(issues_df["age_days"])
        features.append(sev.reshape(-1, 1))
        features.append(eff.reshape(-1, 1))
        features.append(age.reshape(-1, 1))
    else:
        # Normalize issue age (days since creation)
        age = normalize(issues_df["age_days"])
        features.append(age.reshape(-1, 1))

    return np.hstack(features)


def build_dev_rank_map(issues_df: pd.DataFrame):
    """
    Construct a developer ranking map based on issue fixing order.

    Logic:
        - Only considers issues that have a non-null "closed_at" timestamp.
        - Sorts these issues by closure time in ascending order.
        - Assigns ranks incrementally (1 = earliest fixed issue).

    Args:
        issues_df (pd.DataFrame): DataFrame containing at least "closed_at" column.

    Returns:
        dict: Mapping {issue_index: developer_rank}.
    """
    fixed = issues_df.dropna(subset=["closed_at"]).copy()   # 只保留已经被修复（有关闭时间）的 issue
    fixed = fixed.sort_values("closed_at")   # 按照修复时间从早到晚排序
    rank_map = {}
    for rank, idx in enumerate(fixed.index, start=1):    # 给最早修复的 issue rank=1，下一个 rank=2，依此类推
        rank_map[idx] = rank
    return rank_map    # {行索引: 开发者修复顺序}


