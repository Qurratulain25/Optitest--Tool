# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Remove URLs and unwanted characters
    df = df.replace(r'http\S+', '', regex=True)
    df = df.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    # Standardize the 'label' column if exists
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower().str.strip()

    # Drop rows that are completely empty
    df = df.dropna(how='all')

    # Fill numeric columns with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().all():
            df = df.drop(columns=[col])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Fill missing for categorical columns
    non_label_cols = [col for col in df.columns if col != 'label']
    categorical_cols = df[non_label_cols].select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("missing")

    # One-hot encode
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols)

    # Normalize all columns except 'label'
    criteria_cols = [col for col in df.columns if col != 'label']
    scaler = MinMaxScaler()
    df[criteria_cols] = scaler.fit_transform(df[criteria_cols])

    return df
