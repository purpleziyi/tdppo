import requests
import pandas as pd
import numpy as np

# ------------------------
# GitHub Issues Loader
# ------------------------
def load_github_issues(repo: str, max_issues: int = 1000) -> pd.DataFrame:
    """
    Fetch issues from GitHub API with pagination.

    Args:
        repo (str): Repository name, e.g., "aws/aws-sdk-java-v2".
        max_issues (int): Maximum number of issues to fetch (default=1000).

    Returns:
        pd.DataFrame: Issues dataframe with standardized columns.
    """
    url = f"https://api.github.com/repos/{repo}/issues"
    per_page = 100  # GitHub API max
    all_issues = []
    page = 1

    while True:
        params = {"state": "all", "per_page": per_page, "page": page}
        r = requests.get(url, params=params)
        data = r.json()    # Convert JSON response to Python list of dicts

        # if page > 1: break    # 待删！！！

        if not isinstance(data, list):
            print("GitHub API error:", data)  # 打印出来调试
            break

        # stop if no more issues
        if not data or len(all_issues) >= max_issues:
            break

        # Convert issues into structured rows for a DataFrame
        for issue in data:
            if "pull_request" in issue:  # skip PRs
                continue

            # labels_field = issue.get("labels", [])
            # label_names = []
            # # Case 1: when labels is a list
            # if isinstance(labels_field, list):
            #     for lbl in labels_field:
            #         if isinstance(lbl, dict):
            #             name = lbl.get("name")
            #             if name:
            #                 label_names.append(name)
            # # Case 2: when labels is string
            # elif isinstance(labels_field, str):
            #     label_names.append(labels_field)
            # labels_joined = ";".join(label_names)

            all_issues.append({
                "issue_id": issue.get("id"),
                "title": issue.get("title", ""),
                "labels": ";".join([l.get("name", "") for l in issue.get("labels", [])]),
                "state": issue.get("state", ""),
                "created_at": issue.get("created_at"),
                "closed_at": issue.get("closed_at"),
                "text": (issue.get("title") or "") + " " + (issue.get("body") or ""),
                "age_days": (
                    (pd.to_datetime(issue.get("closed_at") or pd.Timestamp.utcnow(), utc=True)
                     - pd.to_datetime(issue.get("created_at"), utc=True)).days
                )
            })
        page += 1

    return pd.DataFrame(all_issues)


# ------------------------
# SonarQube Issues Loader
# ------------------------
def load_sonarcloud_issues(project_key: str, max_issues: int = 2000, status: str = "OPEN") -> pd.DataFrame:
    """
    Fetch issues from SonarCloud API with pagination.

    Args:
        project_key (str): SonarCloud project key, e.g., "aws_aws-sdk-java-v2".
        max_issues (int): Maximum number of issues to fetch (default=2000).
        status (str): Issue status to fetch, default "OPEN".

    Returns:
        pd.DataFrame: Issues dataframe with standardized columns.
    """
    url = "https://sonarcloud.io/api/issues/search"
    ps = 500  # SonarCloud max page size
    all_issues = []
    page = 1

    while True:
        params = {
            "componentKeys": project_key,
            "statuses": status,
            "ps": ps,
            "p": page
        }
        r = requests.get(url, params=params)
        data = r.json()
        issues = data.get("issues", [])

        # stop if no more issues
        if not issues or len(all_issues) >= max_issues:
            break

        for issue in issues:
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

            all_issues.append({
                "issue_id": issue.get("key"),
                "title": issue.get("message"),
                "labels": ";".join(issue.get("tags", [])),
                "state": status.lower(),
                "created_at": issue.get("creationDate"),
                "closed_at": None,  # SonarCloud "OPEN" issues usually no closed date
                "severity": (impact_severity or issue.get("severity", "info")),
                "effort_minutes": minutes,
                "text": issue.get("message", ""),
                "age_days": (
                    (pd.to_datetime(pd.Timestamp.utcnow(), utc=True) - pd.to_datetime(issue.get("creationDate"), utc=True)).days
                )
            })
        page += 1

    return pd.DataFrame(all_issues)


def build_monthly_snapshots(issues_df: pd.DataFrame):
    """
    Build monthly snapshots of open issues.

    Args:
        issues_df (pd.DataFrame): DataFrame with at least
            - "created_at" (timestamp string)
            - "closed_at" (timestamp string or None)

    Returns:
        dict: {month_start_date (pd.Timestamp): DataFrame of open issues}
    """
    df = issues_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")

    # time range
    start_date = df["created_at"].min().to_period("M").to_timestamp().tz_localize("UTC")  # the earliest issue creation time

    if df["closed_at"].notna().any():
        # there is at least one closed_at-value
        end_date = df["closed_at"].max().to_period("M").to_timestamp().tz_localize("UTC")
    else:
        # all is NaT，use lastest creation time as end_date
        end_date = df["created_at"].max().to_period("M").to_timestamp().tz_localize("UTC")


    snapshots = {}
    month = start_date

    while month <= end_date:
        # 本月 1 号的 backlog = 当时仍然 open 的 issues
        mask = (df["created_at"] < month) & (
                df["closed_at"].isna() | (df["closed_at"] > month)
        )
        snapshots[month] = df.loc[mask].copy()
        month = (month + pd.offsets.MonthBegin(1))

    return snapshots



def build_monthly_snapshots_sonar(issues_df: pd.DataFrame):
    """
    Build monthly snapshots for SonarQube issues (ignores closure).
    Each month contains all issues created before that month.

    Args:
        issues_df (pd.DataFrame): Must contain "creationDate" column.

    Returns:
        dict: {month_start_date (pd.Timestamp): DataFrame of issues up to that month}
    """
    df = issues_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # time range：最早的创建时间到最晚的创建时间
    start_date = df["created_at"].min().to_period("M").to_timestamp().tz_localize("UTC")
    end_date = df["created_at"].max().to_period("M").to_timestamp().tz_localize("UTC")

    snapshots = {}
    month = start_date

    while month <= end_date:
        # backlog = 截止到该月一号之前创建的所有 issue
        mask = df["created_at"] < month
        snapshots[month] = df.loc[mask].copy()
        month = month + pd.offsets.MonthBegin(1)

    return snapshots