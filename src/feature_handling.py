import pandas as pd


def encode_string_features(df: pd.DataFrame, custom_mappings: dict | None = None) -> pd.DataFrame:
    """Converts string/categorical columns into integer representations.

    Parameters:
    -----------
    df : pd.DataFrame  Input DataFrame.custom_mappings : dict, optional
        Dictionary specifying explicit ordinal mappings for specific columns,
        e.g., {'Range_Anxiety_Level': {'Low': 0, 'Medium': 1, 'High': 2}}

    Returns:
    --------
    Transformed DataFrame with integer-encoded categorical features.
    """
    df = df.copy()

    if custom_mappings:
        for col, mapping in custom_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

    # Automatically label-encode remaining object/string columns if needed
    string_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in string_cols:
        df[col] = df[col].astype('category').cat.codes

    return df