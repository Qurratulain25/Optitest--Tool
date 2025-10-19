"""
fahp_method.py
================

This script implements the Fuzzy Analytical Hierarchy Process (FAHP)
selection method described in the OptiTest⁺ paper.  It loads each
dataset from the repository, derives four performance criteria
(response time, throughput, latency and network load) using the
``mapping.py`` helper, computes FAHP weights from a set of linguistic
preferences and ranks test cases accordingly.  The top portion of the
suite is retained based on the desired reduction ratio.  Finally,
metrics for reduction, coverage and execution‐time reduction are
reported.

The goal of this module is to provide a standalone implementation of
the FAHP approach so it can be run separately from other baselines.

Usage
-----

Run this file from the root of the project with Python.  It will
process all Excel/CSV datasets in ``Optitest--Tool-main/Dataset`` and
print a summary table.  Adjust ``REDUCTION_RATIO`` to change the
percentage of test cases retained.

"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# To evaluate statistical significance you could import scipy.stats,
# but this module focuses on the selection logic and metrics only.

# Append the repository path so we can import helper modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "Optitest--Tool-main")
sys.path.append(REPO_PATH)

from fahp import calculate_fahp  # type: ignore
from mapping import map_criteria  # type: ignore


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load a dataset and derive performance criteria.

    Parameters
    ----------
    filepath: str
        Path to an Excel (.xlsx) or CSV (.csv) file containing raw
        JMeter results.  The file must include at least the following
        columns: ``elapsed``, ``sentBytes``, ``Latency``, ``allThreads``
        and ``label``.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with additional columns for ``Response_Time``,
        ``Throughput``, ``Latency`` and ``Network_Load``.  All numeric
        columns are normalised into the [0,1] range via min–max
        scaling.  Missing values are imputed with column means.
    criteria: list of str
        Names of the derived criteria present in the returned
        dataframe.
    """
    # Read the input file (Excel or CSV)
    if filepath.lower().endswith(".xlsx"):
        raw_df = pd.read_excel(filepath)
    else:
        raw_df = pd.read_csv(filepath)
    # Derive criteria using helper.  The helper prints a summary.
    df, criteria = map_criteria(raw_df)
    # Ensure numeric types and fill missing values
    for col in criteria:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            # Drop entirely missing columns
            df = df.drop(columns=[col])
            criteria.remove(col)
        else:
            df[col] = df[col].fillna(df[col].mean())
    # Normalise numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    return df, criteria


def compute_global_scores(df: pd.DataFrame, criteria: List[str], weights: np.ndarray) -> pd.Series:
    """Compute FAHP global score for each test case.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing criteria columns.
    criteria: list of str
        Names of criteria columns in ``df``.  Order must match
        ``weights``.
    weights: numpy.ndarray
        Normalised weight vector summing to 1.

    Returns
    -------
    pandas.Series
        Weighted sum of criteria values for each row.
    """
    assert len(criteria) == len(weights)
    values = df[criteria].values
    scores = values.dot(weights)
    return pd.Series(scores, index=df.index)


def select_tests_fahp(df: pd.DataFrame, scores: pd.Series, reduction_ratio: float) -> pd.Index:
    """Select a subset of tests using FAHP scores.

    The tests are sorted in descending order of their scores.  A
    fraction of the suite defined by ``reduction_ratio`` is removed
    (e.g. 0.35 corresponds to a 35 % reduction).

    Returns
    -------
    pandas.Index
        Indices of the selected tests.
    """
    total = len(df)
    retain_count = int(round(total * (1.0 - reduction_ratio)))
    sorted_indices = scores.sort_values(ascending=False).index
    return sorted_indices[:retain_count]


def evaluate_selection(df: pd.DataFrame, selected_indices: pd.Index) -> dict:
    """Calculate reduction, coverage and time reduction metrics.

    * Reduction (%) is the percentage of tests removed.
    * Coverage (%) is the proportion of unique labels retained.
    * Time reduction (%) is computed from the sum of ``elapsed`` times.

    Returns
    -------
    dict
        Dictionary with keys ``reduction_pct``, ``coverage_pct`` and
        ``time_reduction_pct``.
    """
    total_count = len(df)
    selected_count = len(selected_indices)
    reduction_pct = (1.0 - selected_count / total_count) * 100.0
    # Coverage: unique labels retained
    if 'label' in df.columns:
        total_labels = df['label'].nunique()
        selected_labels = df.loc[selected_indices, 'label'].nunique()
        coverage_pct = (selected_labels / total_labels) * 100.0 if total_labels > 0 else 100.0
    else:
        coverage_pct = 100.0
    # Time reduction: sum of elapsed times
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


def run_fahp_on_datasets(datasets_dir: str, reduction_ratio: float = 0.35) -> pd.DataFrame:
    """Execute FAHP selection across all datasets.

    Parameters
    ----------
    datasets_dir: str
        Directory containing the datasets to process.
    reduction_ratio: float
        Fraction of the suite to remove (0.35 for 35 % reduction).

    Returns
    -------
    pandas.DataFrame
        Summary of metrics for each dataset processed.
    """
    results = []
    # Define linguistic preferences for FAHP.  Response Time is
    # strongly more important, Throughput fairly more important,
    # Latency weakly more important and Network Load equally important.
    user_weights = {
        'Throughput': 'fairly_more_important',
        'Latency': 'weakly_more_important',
        'Response_Time': 'strongly_more_important',
        'Network_Load': 'equally_important',
    }
    criteria_order = ['Throughput', 'Latency', 'Response_Time', 'Network_Load']
    fahp_weights, _ = calculate_fahp(user_weights, criteria_order)
    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.lower().endswith((".xlsx", ".csv")):
            continue
        filepath = os.path.join(datasets_dir, filename)
        df, criteria = load_and_preprocess(filepath)
        # Align criteria with weight order
        present_criteria = [c for c in criteria_order if c in df.columns]
        weight_mask = [criteria_order.index(c) for c in present_criteria]
        weights = fahp_weights[weight_mask]
        weights = weights / weights.sum()  # normalise
        scores = compute_global_scores(df, present_criteria, weights)
        selected = select_tests_fahp(df, scores, reduction_ratio)
        metrics = evaluate_selection(df, selected)
        results.append({
            'Dataset': filename,
            'Method': 'FAHP',
            **metrics,
        })
    return pd.DataFrame(results)


def main() -> None:
    """Run FAHP on all datasets and print results."""
    datasets_dir = os.path.join(REPO_PATH, 'Dataset')
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")
    results = run_fahp_on_datasets(datasets_dir, reduction_ratio=0.35)
    print(results.to_string(index=False))
    # Print averages across datasets
    print("\nAverage reduction: {:.2f}%".format(results['reduction_pct'].mean()))
    print("Average coverage: {:.2f}%".format(results['coverage_pct'].mean()))
    print("Average time reduction: {:.2f}%".format(results['time_reduction_pct'].mean()))


if __name__ == '__main__':
    main()