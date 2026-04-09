import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ForecastActionsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = 'http://127.0.0.1:8000/forecast';

  generateDailyForecast(channel: string = 'Choice'): Observable<any> {
    return this.http.post(`${this.baseUrl}/daily`, { channel });
  }
}