# mapping.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import ALTERNATIVE_MAPPING

def map_criteria(df):
    if 'elapsed' in df.columns:
        df['Response_Time'] = df['elapsed']
    else:
        df['Response_Time'] = np.nan

    if 'sentBytes' in df.columns and 'elapsed' in df.columns:
        df['Throughput'] = df['sentBytes'] / df['elapsed'].replace(0, np.nan)
    else:
        df['Throughput'] = np.nan

    if 'Latency' in df.columns:
        df['Latency'] = df['Latency']
    else:
        df['Latency'] = np.nan

    if 'sentBytes' in df.columns and 'allThreads' in df.columns:
        df['Network_Load'] = df['sentBytes'] * df['allThreads']
    else:
        df['Network_Load'] = np.nan

    mapped_criteria = ['Response_Time', 'Throughput', 'Latency', 'Network_Load']
    print("\nMapped Criteria Summary:")
    print("Response_Time: elapsed")
    print("Throughput: sentBytes / elapsed")
    print("Latency: Latency")
    print("Network_Load: sentBytes * allThreads")
    return df, mapped_criteria

def map_alternatives(df, alt_col, chosen_alts):
    col_found = None
    for col in df.columns:
        if col.strip().lower() == alt_col.strip().lower():
            col_found = col
            break
    if col_found is None:
        df[alt_col] = "Other"
        col_found = alt_col

    def map_row(text):
        text_lower = str(text).lower()
        for std_alt in chosen_alts:
            keywords = ALTERNATIVE_MAPPING.get(std_alt, [])
            for kw in keywords:
                if kw in text_lower:
                    return std_alt
        return "Other"
    
    df[col_found] = df[col_found].apply(map_row)
    return df

def renormalize_criteria(df, criteria):
    scaler = MinMaxScaler()
    df[criteria] = scaler.fit_transform(df[criteria])
    return df
