import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def handle_missing_values(df: pd.DataFrame, threshold: float = 0.05, discrete_threshold: int = 20) -> pd.DataFrame:
    """Handles missing data based on column-level missingness thresholds.

    - For columns with > threshold missingness: drops rows with NaNs in these columns.
    - For continuous numeric columns: imputes with median (robust to outliers).
    - For discrete/integer numeric columns: imputes with median and rounds to integer.
    - For categorical fields: imputes with mode.
    """
    df = df.copy()

    # 1. Drop rows for columns exceeding the missingness threshold
    missing_ratios = df.isna().mean()
    high_missing_cols = missing_ratios[missing_ratios > threshold].index

    if not high_missing_cols.empty:
        df = df.dropna(subset=high_missing_cols)

    # 2. Recalculate missing columns after row dropping
    remaining_missing_cols = df.columns[df.isna().any()]

    for col in remaining_missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            non_na = df[col].dropna()
            is_integer_like = (non_na % 1 == 0).all()
            is_discrete = is_integer_like and (
                non_na.nunique() <= discrete_threshold
            )

            if is_discrete:
                # Use median rounded to integer for discrete/ordinal ratings
                fill_val = round(df[col].median())
                df[col] = df[col].fillna(fill_val).astype("Int64")
            else:
                # Use median for continuous variables to avoid outlier sensitivity
                df[col] = df[col].fillna(df[col].median())
        else:
            mode_series = df[col].mode()
            fill_val = (
                mode_series.iloc[0] if not mode_series.empty else "Unknown"
            )
            df[col] = df[col].fillna(fill_val)

    return df




def plot_distributions(
    df: pd.DataFrame, cols: list | None, discrete_threshold: int = 10
):
    """Generically plots bar charts for discrete/categorical variables

    and histogram + KDE plots for continuous numeric variables.
    """
    if cols is None:
        cols = df.columns.tolist()

    valid_cols = [col for col in cols if col in df.columns]
    num_plots = len(valid_cols)

    if num_plots == 0:
        return

    # Dynamically determine grid dimensions (2 columns wide)
    ncols = 2 if num_plots > 1 else 1
    nrows = int(np.ceil(num_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_discrete = is_numeric and (df[col].nunique() <= discrete_threshold)

        if not is_numeric or is_discrete:
            # Bar plot for categorical or low-cardinality discrete fields
            df[col].value_counts().sort_index().plot(
                kind="bar", ax=ax, color="skyblue", edgecolor="black"
            )
            ax.set_title(f"Frequency of {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
        else:
            # Histogram + KDE for continuous numeric fields
            sns.histplot(
                data=df,
                x=col,
                kde=True,
                ax=ax,
                color="skyblue",
                edgecolor="black",
            )
            ax.set_title(f"Distribution of {col}")

    # Remove extra, unused axes from grid
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def remove_outliers_by_zscore(df: pd.DataFrame,  cols: list | None, threshold: float = 3.0) -> pd.DataFrame:
    """Removes rows containing Z-score outliers from numeric columns.

    Parameters:
    - df: Input DataFrame.
    - threshold: Cutoff for absolute Z-score (default is 3.0).
    - cols: Specific numeric columns to evaluate. If None, evaluates all numeric
    columns.

    Returns:
    - Cleaned DataFrame with outlier rows filtered out.
    """
    df_clean = df.copy()

    if cols is None:
        cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if not cols:
        return df_clean

    mean = df_clean[cols].mean()
    std = df_clean[cols].std(ddof=0)

    # Prevent division by zero for zero-variance columns
    std_adjusted = std.replace(0, np.nan)

    z_scores = (df_clean[cols] - mean).abs() / std_adjusted

    # Zero-variance columns will have NaN Z-scores; treat them as non-outliers
    z_scores = z_scores.fillna(0)

    # Keep rows where all specified columns satisfy |Z| < threshold
    mask = (z_scores < threshold).all(axis=1)

    return df_clean[mask].reset_index(drop=True)


