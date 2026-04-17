import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  ForecastHistoryItem,
  ForecastIntervalHistoryItem
} from '../models/system-summary.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ForecastHistoryService {
  private readonly http = inject(HttpClient);
  private readonly historyUrl = `${environment.apiUrl}/forecast/history`;
  private readonly intervalHistoryUrl = `${environment.apiUrl}/forecast/history/intervals`;

  getHistory(channel: string = 'Choice', limit: number = 10): Observable<ForecastHistoryItem[]> {
    const params = new HttpParams()
      .set('channel', channel)
      .set('limit', limit);

    return this.http.get<ForecastHistoryItem[]>(this.historyUrl, { params });
  }

  getIntervalHistory(
    channel: string = 'Choice',
    forecastDate?: string | null,
    limit: number = 2000
  ): Observable<ForecastIntervalHistoryItem[]> {
    let params = new HttpParams()
      .set('channel', channel)
      .set('limit', limit);

    if (forecastDate) {
      params = params.set('forecast_date', forecastDate);
    }

    return this.http.get<ForecastIntervalHistoryItem[]>(this.intervalHistoryUrl, { params });
  }
}
