import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import { QualityReport } from '../models/quality.model';

@Injectable({ providedIn: 'root' })
export class QualityService {
  private readonly apiUrl = `${environment.apiUrl}/quality/report`;

  constructor(private readonly http: HttpClient) {}

  getQualityReport(): Observable<QualityReport> {
    return this.http.get<QualityReport>(this.apiUrl);
  }
}
