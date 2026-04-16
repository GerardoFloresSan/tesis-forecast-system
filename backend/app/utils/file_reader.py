from pathlib import Path

import pandas as pd

from app.utils.normalizer import CANONICAL_COLUMN_ALIASES, normalize_column_name


EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}
CSV_EXTENSIONS = {".csv"}


def _score_sheet(df: pd.DataFrame) -> int:
    normalized_columns = {normalize_column_name(col) for col in df.columns}
    score = 0

    for aliases in CANONICAL_COLUMN_ALIASES.values():
        normalized_aliases = {normalize_column_name(alias) for alias in aliases}
        if normalized_columns & normalized_aliases:
            score += 1

    return score


def read_file(file_path: str, preferred_sheet_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in CSV_EXTENSIONS:
        df = pd.read_csv(path)
        return df, {
            "file_type": "csv",
            "sheet_used": None,
            "available_sheets": [],
        }

    if suffix not in EXCEL_EXTENSIONS:
        raise ValueError(f"Formato de archivo no soportado: {suffix}")

    workbook = pd.ExcelFile(path)
    available_sheets = workbook.sheet_names

    if not available_sheets:
        raise ValueError("El archivo Excel no contiene hojas legibles.")

    if preferred_sheet_name and preferred_sheet_name in available_sheets:
        selected_sheet = preferred_sheet_name
    else:
        selected_sheet = None
        best_score = -1

        for sheet_name in available_sheets:
            sample_df = pd.read_excel(path, sheet_name=sheet_name, nrows=5)
            score = _score_sheet(sample_df)
            if score > best_score:
                best_score = score
                selected_sheet = sheet_name

        if selected_sheet is None:
            selected_sheet = available_sheets[0]

    df = pd.read_excel(path, sheet_name=selected_sheet)

    return df, {
        "file_type": "excel",
        "sheet_used": selected_sheet,
        "available_sheets": available_sheets,
    }