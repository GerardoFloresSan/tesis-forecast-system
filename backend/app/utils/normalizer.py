import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed

def treat_nulls(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    nulls_before = int(df.isnull().sum().sum())
    df = df.fillna(0)
    return df, nulls_before