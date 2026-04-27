import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import { ExternalVariable } from '../models/external-variable.model';

@Injectable({ providedIn: 'root' })
export class ExternalVariablesService {
  private readonly apiUrl = `${environment.apiUrl}/external-variables/`;

  constructor(private readonly http: HttpClient) {}

  getExternalVariables(): Observable<ExternalVariable[]> {
    return this.http.get<ExternalVariable[]>(this.apiUrl);
  }
}
