import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  ForecastBatchResponse,
  ForecastMonthlyResponse
} from '../models/system-summary.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ForecastActionsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/forecast`;

  generateDailyForecast(channel: string = 'Choice'): Observable<ForecastBatchResponse> {
    return this.http.post<ForecastBatchResponse>(`${this.baseUrl}/daily`, { channel });
  }

  generateMonthlyForecast(
    channel: string,
    startDate: string,
    endDate: string
  ): Observable<ForecastMonthlyResponse> {
    return this.http.post<ForecastMonthlyResponse>(`${this.baseUrl}/monthly`, {
      channel,
      start_date: startDate,
      end_date: endDate
    });
  }
}
