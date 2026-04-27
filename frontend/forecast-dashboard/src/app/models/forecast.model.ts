export interface ForecastInterval {
  id: number;
  forecast_run_id: number;
  channel: string;
  forecast_date: string;
  forecast_datetime: string;
  interval_time: string;
  slot_index: number;
  shift_label: string;
  predicted_value: number;

  aht?: number | null;
  required_agents?: number | null;

  model_version?: string | null;
  created_at: string;
}