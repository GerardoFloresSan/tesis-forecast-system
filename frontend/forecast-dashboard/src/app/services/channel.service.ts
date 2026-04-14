import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ChannelService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = `${environment.apiUrl}/forecast/channels`;

  getChannels(): Observable<string[]> {
    return this.http.get<string[]>(this.apiUrl);
  }
}
