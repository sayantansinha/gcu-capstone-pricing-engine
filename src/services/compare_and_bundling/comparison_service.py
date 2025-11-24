from typing import List

import pandas as pd


def build_compare_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a comparison table (titles as columns; key fields as rows).

    Expected columns:
        - title_name
        - predicted_price
        - region
        - platform
        - release_year
        - genres
    """
    if predictions_df.empty:
        return predictions_df

    cols_for_compare: List[str] = [
        "title_name",
        "predicted_price",
        "region",
        "platform",
        "release_year",
        "genres",
    ]
    present_cols = [c for c in cols_for_compare if c in predictions_df.columns]
    df = predictions_df[present_cols].copy()

    df_rows = []
    row_labels = {
        "predicted_price": "Predicted Price",
        "region": "Region",
        "platform": "Platform",
        "release_year": "Release Year",
        "genres": "Genres",
    }

    title_names = (
        df["title_name"].tolist()
        if "title_name" in df.columns
        else [f"Title {i + 1}" for i in range(len(df))]
    )

    for field, label in row_labels.items():
        if field not in df.columns:
            continue
        row = {"Metric": label}
        for i, title in enumerate(title_names):
            row[title] = df.iloc[i][field]
        df_rows.append(row)

    compare_df = pd.DataFrame(df_rows)
    return compare_df
