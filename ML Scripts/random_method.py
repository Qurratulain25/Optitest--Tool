"""
random_method.py
================

This script implements a simple random sampling baseline for the
OptiTest⁺ datasets.  It loads each dataset, derives performance
criteria via ``mapping.py`` and then randomly selects the same
proportion of test cases as the FAHP method (retaining 65 %,
removing 35 %).  After selection, the script reports the reduction
ratio, coverage of unique labels and reduction in total execution
time.

Running this file provides a standalone baseline for comparison
against more sophisticated prioritisation techniques.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# Append the repository path to import helper modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "Optitest--Tool-main")
sys.path.append(REPO_PATH)

from mapping import map_criteria  # type: ignore


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load and normalise a dataset (shared with other methods)."""
    if filepath.lower().endswith(".xlsx"):
        raw_df = pd.read_excel(filepath)
    else:
        raw_df = pd.read_csv(filepath)
    df, criteria = map_criteria(raw_df)
    # Ensure numeric types and fill missing values
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
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    return df, criteria


def select_tests_random(df: pd.DataFrame, retain_count: int, seed: int = 0) -> pd.Index:
    """Randomly select a fixed number of test cases."""
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(df.index, size=retain_count, replace=False)
    return pd.Index(selected_indices)


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


def run_random_on_datasets(datasets_dir: str, reduction_ratio: float = 0.35) -> pd.DataFrame:
    """Execute random selection on all datasets and return a summary."""
    results = []
    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.lower().endswith((".xlsx", ".csv")):
            continue
        filepath = os.path.join(datasets_dir, filename)
        df, _ = load_and_preprocess(filepath)
        total = len(df)
        retain_count = int(round(total * (1.0 - reduction_ratio)))
        selected = select_tests_random(df, retain_count, seed=42)
        metrics = evaluate_selection(df, selected)
        results.append({
            'Dataset': filename,
            'Method': 'Random',
            **metrics,
        })
    return pd.DataFrame(results)


def main() -> None:
    datasets_dir = os.path.join(REPO_PATH, 'Dataset')
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")
    results = run_random_on_datasets(datasets_dir, reduction_ratio=0.35)
    print(results.to_string(index=False))
    print("\nAverage reduction: {:.2f}%".format(results['reduction_pct'].mean()))
    print("Average coverage: {:.2f}%".format(results['coverage_pct'].mean()))
    print("Average time reduction: {:.2f}%".format(results['time_reduction_pct'].mean()))


if __name__ == '__main__':
    main()