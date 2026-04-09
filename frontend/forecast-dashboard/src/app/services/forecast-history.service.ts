import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ForecastHistoryItem } from '../models/system-summary.model';

@Injectable({
  providedIn: 'root'
})
export class ForecastHistoryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000/forecast/history';

  getHistory(channel: string = 'Choice', limit: number = 10): Observable<ForecastHistoryItem[]> {
    return this.http.get<ForecastHistoryItem[]>(
      `${this.apiUrl}?channel=${channel}&limit=${limit}`
    );
  }
}