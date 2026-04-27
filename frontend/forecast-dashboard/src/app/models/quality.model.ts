export interface QualityDateRange {
  start_date: string | null;
  end_date: string | null;
  total_days: number;
}

export interface QualityDuplicateKeys {
  duplicate_groups: number;
  duplicate_records: number;
  sample: unknown[];
}

export interface QualityIntervals {
  channels: unknown[];
  total_invalid_intervals: number;
  total_missing_intervals: number;
}

export interface QualityDaysWithoutData {
  count: number;
  dates: string[];
  by_channel: Record<string, string[]>;
}

export interface QualitySummary {
  status: string;
  issues: string[];
}

export interface QualityReport {
  total_records: number;
  missing_percentage: number;
  duplicate_percentage: number;
  valid_percentage: number;
  date_range: QualityDateRange;
  detected_channels: string[];
  records_by_channel: Record<string, number>;
  nulls_by_column: Record<string, number>;
  duplicate_keys: QualityDuplicateKeys;
  intervals: QualityIntervals;
  days_without_data: QualityDaysWithoutData;
  summary: QualitySummary;
}

export type QualityRiskStatus = 'Óptimo' | 'Aceptable' | 'Deficiente';
