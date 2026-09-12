import pandas as pd


import pandas as pd


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Handles missing data based on a column-level missingness threshold.

    - For columns with > threshold missingness: drops rows with NaNs in these
    columns.
    - For columns with <= threshold missingness: imputes numeric fields with the
    mean and categorical fields with the mode.
    """
    df = df.copy()

    # Calculate proportion of missing values per column
    missing_ratios = df.isna().mean()

    # 1. Drop rows where missingness in specific columns exceeds threshold
    high_missing_cols = missing_ratios[missing_ratios > threshold].index
    if len(high_missing_cols) > 0:
        df = df.dropna(subset=high_missing_cols)

    # 2. Impute remaining columns below or equal to threshold
    low_missing_cols = missing_ratios[(missing_ratios > 0) & (missing_ratios <= threshold)].index

    for col in low_missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df