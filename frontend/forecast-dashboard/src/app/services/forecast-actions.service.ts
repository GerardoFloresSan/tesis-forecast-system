import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import {
  ForecastBatchResponse,
  ForecastMonthlyResponse,
  ForecastMonthlyStatusResponse
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

  getMonthlyForecastStatus(
    channel: string,
    startDate: string,
    endDate: string
  ): Observable<ForecastMonthlyStatusResponse> {
    const params = new HttpParams()
      .set('channel', channel)
      .set('start_date', startDate)
      .set('end_date', endDate);

    return this.http.get<ForecastMonthlyStatusResponse>(
      `${this.baseUrl}/monthly/status`,
      { params }
    );
  }
}