"""
pareto_method.py
=================

This script implements a Pareto‑front based selection strategy for
performance test case prioritisation.  Each test case is evaluated
along multiple criteria (normalised response time, throughput,
latency and network load).  A non‑dominated sorting algorithm
identifies the Pareto fronts, and tests on the first front are
preferred.  If fewer than the desired number of tests exist on the
first front, subsequent fronts are taken in order until the quota
is met.  Within each front, tests are sorted by the sum of
normalised criteria (smaller sum indicates a more balanced test).

Although Pareto ranking can identify high‑impact tests, it may
prioritise only bottleneck cases, resulting in limited coverage and
modest time reduction.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "Optitest--Tool-main")
sys.path.append(REPO_PATH)

from mapping import map_criteria  # type: ignore


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset and derive performance criteria (shared).

    This function normalises all numeric columns into the [0,1]
    interval and handles missing values.
    """
    if filepath.lower().endswith(".xlsx"):
        raw_df = pd.read_excel(filepath)
    else:
        raw_df = pd.read_csv(filepath)
    df, criteria = map_criteria(raw_df)
    for col in criteria:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df = df.drop(columns=[col])
            criteria.remove(col)
        else:
            df[col] = df[col].fillna(df[col].mean())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    return df, criteria


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if vector a Pareto‑dominates vector b."""
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(values: np.ndarray) -> List[List[int]]:
    """Perform non‑dominated sorting on a set of points.

    Parameters
    ----------
    values: np.ndarray
        Array of shape (n_samples, n_criteria).  Each row represents
        the performance metrics of a test case.

    Returns
    -------
    list of lists
        A list where each element is a list of indices belonging to
        the same Pareto front.  The first element corresponds to the
        non‑dominated front (rank 1).
    """
    n = values.shape[0]
    S = [[] for _ in range(n)]  # sets of dominated indices
    domination_count = [0] * n  # number of dominating points for each i
    fronts: List[List[int]] = []
    front = []
    # Determine domination relationships
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(values[i], values[j]):
                S[i].append(j)
            elif _dominates(values[j], values[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            front.append(i)
    fronts.append(front)
    # Generate subsequent fronts
    current_front = front
    while current_front:
        next_front: List[int] = []
        for i in current_front:
            for j in S[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        if current_front:
            fronts.append(current_front)
    return fronts


def select_tests_pareto(df: pd.DataFrame, criteria: List[str], retain_count: int) -> pd.Index:
    """Select tests using Pareto ranking and a simple tie‑breaker.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataset containing normalised criteria columns.
    criteria: list of str
        Names of the criteria columns to use for non‑dominated sorting.
    retain_count: int
        Number of test cases to retain.

    Returns
    -------
    pandas.Index
        Indices of selected test cases.
    """
    # Extract criteria matrix (minimisation for all criteria)
    values = df[criteria].values
    fronts = non_dominated_sort(values)
    selected = []
    for front in fronts:
        if len(selected) >= retain_count:
            break
        # Sort within the front by the sum of criteria (smaller sums first)
        front_sorted = sorted(front, key=lambda idx: values[idx].sum())
        needed = retain_count - len(selected)
        selected += front_sorted[:needed]
    return pd.Index(selected[:retain_count])


def evaluate_selection(df: pd.DataFrame, selected_indices: pd.Index) -> dict:
    """Compute reduction, coverage and time reduction metrics."""
    total_count = len(df)
    selected_count = len(selected_indices)
    reduction_pct = (1.0 - selected_count / total_count) * 100.0
    if 'label' in df.columns:
        total_labels = df['label'].nunique()
        selected_labels = df.loc[selected_indices, 'label'].nunique()
        coverage_pct = (selected_labels / total_labels) * 100.0 if total_labels > 0 else 100.0
    else:
        coverage_pct = 100.0
    if 'elapsed' in df.columns:
        total_elapsed = df['elapsed'].sum()
        selected_elapsed = df.loc[selected_indices, 'elapsed'].sum()
        time_reduction_pct = (1.0 - selected_elapsed / total_elapsed) * 100.0 if total_elapsed > 0 else 0.0
    else:
        time_reduction_pct = 0.0
    return {
        'reduction_pct': reduction_pct,
        'coverage_pct': coverage_pct,
        'time_reduction_pct': time_reduction_pct,
    }


def run_pareto_on_datasets(datasets_dir: str, reduction_ratio: float = 0.35) -> pd.DataFrame:
    """Apply Pareto selection across all datasets."""
    results = []
    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.lower().endswith((".xlsx", ".csv")):
            continue
        filepath = os.path.join(datasets_dir, filename)
        df, criteria = load_and_preprocess(filepath)
        total = len(df)
        retain_count = int(round(total * (1.0 - reduction_ratio)))
        selected = select_tests_pareto(df, criteria, retain_count)
        metrics = evaluate_selection(df, selected)
        results.append({
            'Dataset': filename,
            'Method': 'Pareto',
            **metrics,
        })
    return pd.DataFrame(results)


def main() -> None:
    datasets_dir = os.path.join(REPO_PATH, 'Dataset')
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")
    results = run_pareto_on_datasets(datasets_dir, reduction_ratio=0.35)
    print(results.to_string(index=False))
    print("\nAverage reduction: {:.2f}%".format(results['reduction_pct'].mean()))
    print("Average coverage: {:.2f}%".format(results['coverage_pct'].mean()))
    print("Average time reduction: {:.2f}%".format(results['time_reduction_pct'].mean()))


if __name__ == '__main__':
    main()