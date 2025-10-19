"""
coverage_method.py
===================

This script implements a simple coverage‑oriented heuristic for the
OptiTest⁺ datasets.  The heuristic aims to maximise the number of
unique request labels retained when selecting a reduced subset of
tests.  It works in two phases: first, it selects the test case
with the lowest response time (elapsed time) for each unique label;
then, if additional slots are required to reach the desired
retention ratio, it fills them with the fastest remaining tests.

Coverage is approximated by the proportion of unique labels present
in the selected subset.  While this method typically yields large
time reductions, it may bias heavily toward lightweight scenarios
and thus miss performance bottlenecks.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# Append repository path to import helper modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "Optitest--Tool-main")
sys.path.append(REPO_PATH)

from mapping import map_criteria  # type: ignore


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset and normalise numeric columns (as in other methods)."""
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


def select_tests_coverage(df: pd.DataFrame, retain_count: int, label_col: str = 'label') -> pd.Index:
    """Select tests to maximise unique label coverage.

    Parameters
    ----------
    df: pandas.DataFrame
        The preprocessed dataset.
    retain_count: int
        Number of tests to retain.
    label_col: str
        Name of the column containing request labels.

    Returns
    -------
    pandas.Index
        Indices of the selected test cases.
    """
    selected = []
    if label_col in df.columns:
        unique_labels = df[label_col].dropna().unique().tolist()
        for lbl in unique_labels:
            subset = df[df[label_col] == lbl]
            if subset.empty:
                continue
            # choose the test with minimal elapsed time for this label
            idx = subset['elapsed'].idxmin()
            selected.append(idx)
    # Fill remaining slots with fastest remaining tests
    remaining = [i for i in df.index if i not in selected]
    remaining_sorted = sorted(remaining, key=lambda i: df.loc[i, 'elapsed'])
    additional_needed = max(0, retain_count - len(selected))
    selected += remaining_sorted[:additional_needed]
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


def run_coverage_on_datasets(datasets_dir: str, reduction_ratio: float = 0.35) -> pd.DataFrame:
    """Apply the coverage heuristic across all datasets."""
    results = []
    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.lower().endswith((".xlsx", ".csv")):
            continue
        filepath = os.path.join(datasets_dir, filename)
        df, _ = load_and_preprocess(filepath)
        total = len(df)
        retain_count = int(round(total * (1.0 - reduction_ratio)))
        selected = select_tests_coverage(df, retain_count)
        metrics = evaluate_selection(df, selected)
        results.append({
            'Dataset': filename,
            'Method': 'Coverage',
            **metrics,
        })
    return pd.DataFrame(results)


def main() -> None:
    datasets_dir = os.path.join(REPO_PATH, 'Dataset')
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")
    results = run_coverage_on_datasets(datasets_dir, reduction_ratio=0.35)
    print(results.to_string(index=False))
    print("\nAverage reduction: {:.2f}%".format(results['reduction_pct'].mean()))
    print("Average coverage: {:.2f}%".format(results['coverage_pct'].mean()))
    print("Average time reduction: {:.2f}%".format(results['time_reduction_pct'].mean()))


if __name__ == '__main__':
    main()