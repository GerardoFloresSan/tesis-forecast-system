export interface UploadResponse {
  file_name: string;
  sheet_used: string | null;
  records_original: number;
  duplicates_removed: number;
  nulls_treated: number;
  records_replaced: number;
  records_final: number;
  load_mode: string;
  message: string;
}
