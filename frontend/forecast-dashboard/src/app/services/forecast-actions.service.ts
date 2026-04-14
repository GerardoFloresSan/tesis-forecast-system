import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ForecastActionsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/forecast`;

  generateDailyForecast(channel: string = 'Choice'): Observable<any> {
    return this.http.post(`${this.baseUrl}/daily`, { channel });
  }
}
