import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ForecastBatchResponse } from '../models/system-summary.model';
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
}
