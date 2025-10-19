"""
nsga2_method.py
================

This script implements a simplified NSGA-II selection procedure for
performance test prioritisation.  Similar to the Pareto approach, it
uses non-dominated sorting to assign ranks to test cases.  Within
each front, it computes a crowding distance to favour diversity
across criteria.  Tests are selected from successive fronts until
the desired number of cases is reached, with tie-breaking based on
descending crowding distance.

While NSGA-II is powerful for multi-objective optimisation, it often
prioritises tests with extreme values on one criterion, leading to
large subsets dominated by high-impact cases and potentially lower
overall time reduction compared with FAHP.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "Optitest--Tool-main")
sys.path.append(REPO_PATH)

from mapping import map_criteria  # type: ignore


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset and normalise numeric columns."""
    # Load file
    if filepath.lower().endswith(".xlsx"):
        raw_df = pd.read_excel(filepath)
    else:
        raw_df = pd.read_csv(filepath)

    # Map criteria using the custom mapping function
    df, criteria = map_criteria(raw_df)

    # Clean and convert numeric columns
    for col in criteria:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df = df.drop(columns=[col])
            criteria.remove(col)
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, criteria


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(values: np.ndarray) -> List[List[int]]:
    """Perform non-dominated sorting and return a list of fronts."""
    n = values.shape[0]
    S = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: List[List[int]] = []
    front = []

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


def crowding_distance(values: np.ndarray, front: List[int]) -> np.ndarray:
    """Compute crowding distance for a given front."""
    if not front:
        return np.array([])

    m = values.shape[1]
    distances = np.zeros(len(front))

    # For each criterion
    for k in range(m):
        idxs = sorted(range(len(front)), key=lambda i: values[front[i], k])
        # Boundary points get infinite distance
        distances[idxs[0]] = distances[idxs[-1]] = np.inf
        # Compute distances for interior points
        min_val = values[front[idxs[0]], k]
        max_val = values[front[idxs[-1]], k]
        if max_val > min_val:
            for i in range(1, len(front) - 1):
                prev_val = values[front[idxs[i - 1]], k]
                next_val = values[front[idxs[i + 1]], k]
                distances[idxs[i]] += (next_val - prev_val) / (max_val - min_val)

    return distances


def select_tests_nsga2(df: pd.DataFrame, criteria: List[str], retain_count: int) -> pd.Index:
    """Select tests using non-dominated sorting and crowding distance."""
    values = df[criteria].values
    fronts = non_dominated_sort(values)
    selected: List[int] = []

    for front in fronts:
        if len(selected) >= retain_count:
            break
        distances = crowding_distance(values, front)
        order = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
        front_sorted = [front[i] for i in order]
        needed = retain_count - len(selected)
        selected += front_sorted[:needed]

    return pd.Index(selected[:retain_count])


def evaluate_selection(df: pd.DataFrame, selected_indices: pd.Index) -> dict:
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


def run_nsga2_on_datasets(datasets_dir: str, reduction_ratio: float = 0.35) -> pd.DataFrame:
    """Apply the NSGA-II selection across all datasets."""
    results = []
    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.lower().endswith((".xlsx", ".csv")):
            continue
        filepath = os.path.join(datasets_dir, filename)
        df, criteria = load_and_preprocess(filepath)
        total = len(df)
        retain_count = int(round(total * (1.0 - reduction_ratio)))
        selected = select_tests_nsga2(df, criteria, retain_count)
        metrics = evaluate_selection(df, selected)
        results.append({
            'Dataset': filename,
            'Method': 'NSGA2',
            **metrics,
        })
    return pd.DataFrame(results)


def main() -> None:
    datasets_dir = os.path.join(REPO_PATH, 'Dataset')
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")

    results = run_nsga2_on_datasets(datasets_dir, reduction_ratio=0.35)
    print(results.to_string(index=False))
    print("\nAverage reduction: {:.2f}%".format(results['reduction_pct'].mean()))
    print("Average coverage: {:.2f}%".format(results['coverage_pct'].mean()))
    print("Average time reduction: {:.2f}%".format(results['time_reduction_pct'].mean()))


if __name__ == '__main__':
    main()
