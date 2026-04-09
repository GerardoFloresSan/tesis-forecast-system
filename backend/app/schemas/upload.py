from pydantic import BaseModel

class UploadSummary(BaseModel):
    file_name: str
    records_original: int
    duplicates_removed: int
    nulls_treated: int
    records_final: int
    message: str