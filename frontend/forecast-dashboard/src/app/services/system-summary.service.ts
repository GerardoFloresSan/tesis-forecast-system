import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SystemSummaryResponse } from '../models/system-summary.model'; 

@Injectable({
  providedIn: 'root'
})
export class SystemSummaryService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000/model/system-summary';

  getSummary(channel: string = 'Choice'): Observable<SystemSummaryResponse> {
    return this.http.get<SystemSummaryResponse>(`${this.apiUrl}?channel=${channel}`);
  }
}