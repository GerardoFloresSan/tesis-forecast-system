import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import { UploadResponse } from '../models/upload.model';

@Injectable({ providedIn: 'root' })
export class UploadService {
  private readonly apiUrl = `${environment.apiUrl}/upload/excel`;

  constructor(private readonly http: HttpClient) {}

  uploadFile(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(this.apiUrl, formData);
  }
}
