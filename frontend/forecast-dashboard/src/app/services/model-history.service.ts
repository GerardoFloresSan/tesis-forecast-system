import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { LstmHistoryItem } from '../models/system-summary.model';

@Injectable({
  providedIn: 'root'
})
export class ModelHistoryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000/model/lstm-history';

  getHistory(channel: string = 'Choice', limit: number = 10): Observable<LstmHistoryItem[]> {
    return this.http.get<LstmHistoryItem[]>(
      `${this.apiUrl}?channel=${channel}&limit=${limit}`
    );
  }
}