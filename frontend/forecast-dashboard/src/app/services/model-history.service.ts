import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { LstmHistoryItem } from '../models/system-summary.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ModelHistoryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = `${environment.apiUrl}/model/lstm-history`;

  getHistory(channel: string = 'Choice', limit: number = 10): Observable<LstmHistoryItem[]> {
    return this.http.get<LstmHistoryItem[]>(
      `${this.apiUrl}?channel=${channel}&limit=${limit}`
    );
  }
}
