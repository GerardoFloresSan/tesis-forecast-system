import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { SLAAlertResponse } from '../models/system-summary.model';

@Injectable({
  providedIn: 'root'
})
export class AlertsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/alerts`;

  getActive(channel?: string | null, limit: number = 50): Observable<SLAAlertResponse[]> {
    let params = new HttpParams().set('limit', limit);
    if (channel) {
      params = params.set('channel', channel);
    }

    return this.http.get<SLAAlertResponse[]>(`${this.baseUrl}/active`, { params });
  }

  getHistory(channel?: string | null, limit: number = 100): Observable<SLAAlertResponse[]> {
    let params = new HttpParams().set('limit', limit);
    if (channel) {
      params = params.set('channel', channel);
    }

    return this.http.get<SLAAlertResponse[]>(`${this.baseUrl}/history`, { params });
  }
}