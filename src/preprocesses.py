import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def create_rolling_features(df, window=5):
    for col in df.columns:
        if col not in ['machine_id', 'timestamp', 'label']:
            df[f'{col}_rolling_mean'] = df[col].rolling(window).mean()
    return df