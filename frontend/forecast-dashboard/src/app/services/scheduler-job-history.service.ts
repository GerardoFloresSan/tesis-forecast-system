import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SchedulerJobHistoryItem } from '../models/system-summary.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class SchedulerJobHistoryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = `${environment.apiUrl}/model/scheduler-job-history`;

  getHistory(limit: number = 20): Observable<SchedulerJobHistoryItem[]> {
    return this.http.get<SchedulerJobHistoryItem[]>(`${this.apiUrl}?limit=${limit}`);
  }
}
