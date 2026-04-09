export interface SchedulerJob {
  id: string;
  trigger: string;
  next_run_time: string | null;
}

export interface SchedulerStatus {
  running: boolean;
  jobs: SchedulerJob[];
}

export interface LstmStatus {
  channel: string;
  model_exists: boolean;
  scaler_exists: boolean;
  metadata_exists: boolean;
  metrics_exists: boolean;
}

export interface LstmMetrics {
  channel: string;
  mae: number;
  rmse: number;
  mape: number;
  r2: number;
  train_size: number;
  test_size: number;
  model_path: string;
  scaler_path: string;
  metadata_path: string;
  metrics_path: string;
}

export interface LatestForecast {
  id: number;
  channel: string;
  forecast_date: string;
  predicted_value: number;
  model_version: string | null;
  created_at: string;
}

export interface LatestTrainRun {
  id: number;
  channel: string;
  run_type: string;
  status: string;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  r2: number | null;
  created_at: string;
}

export interface LatestSchedulerJob {
  id: number;
  job_name: string;
  job_type: string;
  channel: string | null;
  status: string;
  action_taken: string;
  message: string | null;
  created_at: string;
}

export interface SystemSummaryResponse {
  channel: string;
  scheduler_status: SchedulerStatus;
  lstm_status: LstmStatus;
  lstm_metrics: LstmMetrics | null;
  latest_forecast: LatestForecast | null;
  latest_train_run: LatestTrainRun | null;
  latest_scheduler_job: LatestSchedulerJob | null;
}

export interface ForecastHistoryItem {
  id: number;
  channel: string;
  forecast_date: string;
  predicted_value: number;
  model_version: string | null;
  created_at: string;
}

export interface LstmHistoryItem {
  id: number;
  channel: string;
  run_type: string;
  status: string;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  r2: number | null;
  train_size: number | null;
  test_size: number | null;
  model_path: string | null;
  scaler_path: string | null;
  metadata_path: string | null;
  metrics_path: string | null;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
}

export interface SchedulerJobHistoryItem {
  id: number;
  job_name: string;
  job_type: string;
  channel: string | null;
  status: string;
  action_taken: string;
  message: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
}