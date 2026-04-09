from pydantic import BaseModel

class QualityReportResponse(BaseModel):
    total_records: int
    missing_percentage: float
    duplicate_percentage: float
    valid_percentage: float