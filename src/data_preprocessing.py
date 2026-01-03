import pandas as pd

def load_and_preprocess_data(path="data/sample_data.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df
