import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { ForecastMonitoringResponse } from '../models/system-summary.model';

@Injectable({
  providedIn: 'root'
})
export class ForecastMonitoringService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/forecast/monitoring`;

  getSummary(channel: string = 'Choice'): Observable<ForecastMonitoringResponse> {
    const params = new HttpParams().set('channel', channel);
    return this.http.get<ForecastMonitoringResponse>(`${this.baseUrl}/summary`, { params });
  }

  getByDate(channel: string, forecastDate: string): Observable<ForecastMonitoringResponse> {
    const params = new HttpParams()
      .set('channel', channel)
      .set('forecast_date', forecastDate);

    return this.http.get<ForecastMonitoringResponse>(`${this.baseUrl}/by-date`, { params });
  }
}