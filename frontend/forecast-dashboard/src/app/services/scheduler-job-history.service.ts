import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SchedulerJobHistoryItem } from '../models/system-summary.model';

@Injectable({
  providedIn: 'root'
})
export class SchedulerJobHistoryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000/model/scheduler-job-history';

  getHistory(limit: number = 20): Observable<SchedulerJobHistoryItem[]> {
    return this.http.get<SchedulerJobHistoryItem[]>(`${this.apiUrl}?limit=${limit}`);
  }
}