import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ForecastHistoryItem } from '../models/system-summary.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ForecastHistoryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = `${environment.apiUrl}/forecast/history`;

  getHistory(channel: string = 'Choice', limit: number = 10): Observable<ForecastHistoryItem[]> {
    return this.http.get<ForecastHistoryItem[]>(
      `${this.apiUrl}?channel=${channel}&limit=${limit}`
    );
  }
}
