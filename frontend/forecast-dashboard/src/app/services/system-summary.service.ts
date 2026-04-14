import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SystemSummaryResponse } from '../models/system-summary.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class SystemSummaryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = `${environment.apiUrl}/model/system-summary`;

  getSummary(channel: string = 'Choice'): Observable<SystemSummaryResponse> {
    return this.http.get<SystemSummaryResponse>(`${this.apiUrl}?channel=${channel}`);
  }
}
