from pydantic import BaseModel


class UploadSummary(BaseModel):
    file_name: str
    sheet_used: str | None = None
    records_original: int
    duplicates_removed: int
    nulls_treated: int
    records_replaced: int = 0
    records_final: int
    load_mode: str = "incremental_replace"
    message: str