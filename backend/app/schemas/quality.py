from pydantic import BaseModel


class DateRangeSummary(BaseModel):
    start_date: str | None
    end_date: str | None
    total_days: int


class DuplicateKeySample(BaseModel):
    interaction_date: str | None
    interval_time: str | None
    channel: str | None
    occurrences: int


class DuplicateKeysSummary(BaseModel):
    duplicate_groups: int
    duplicate_records: int
    sample: list[DuplicateKeySample]


class IntervalDaySample(BaseModel):
    interaction_date: str | None
    intervals: list[str]


class ChannelIntervalSummary(BaseModel):
    channel: str
    cadence_minutes: int | None
    dates_analyzed: int
    invalid_intervals_count: int
    missing_intervals_count: int
    days_with_issues: int
    sample_invalid_intervals: list[IntervalDaySample]
    sample_missing_intervals: list[IntervalDaySample]


class IntervalQualitySummary(BaseModel):
    channels: list[ChannelIntervalSummary]
    total_invalid_intervals: int
    total_missing_intervals: int


class DaysWithoutDataSummary(BaseModel):
    count: int
    dates: list[str]
    by_channel: dict[str, list[str]]


class GeneralSummary(BaseModel):
    status: str
    issues: list[str]


class QualityReportResponse(BaseModel):
    total_records: int
    missing_percentage: float
    duplicate_percentage: float
    valid_percentage: float
    date_range: DateRangeSummary
    detected_channels: list[str]
    records_by_channel: dict[str, int]
    nulls_by_column: dict[str, int]
    duplicate_keys: DuplicateKeysSummary
    intervals: IntervalQualitySummary
    days_without_data: DaysWithoutDataSummary
    summary: GeneralSummary