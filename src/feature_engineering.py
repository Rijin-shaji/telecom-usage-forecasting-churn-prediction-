def add_features(df):
    # Example: Usage per call minute
    df["usage_per_min"] = df["Monthly_Usage"] / (df["Call_Minutes"] + 1)
    return df
