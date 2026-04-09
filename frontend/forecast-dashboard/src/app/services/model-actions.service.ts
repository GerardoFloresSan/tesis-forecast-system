import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ModelActionsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = 'http://127.0.0.1:8000/model';

  trainLstm(channel: string = 'Choice'): Observable<any> {
    return this.http.post(`${this.baseUrl}/train-lstm?channel=${channel}`, {});
  }

  retrainLstm(channel: string = 'Choice'): Observable<any> {
    return this.http.post(`${this.baseUrl}/retrain-lstm?channel=${channel}`, {});
  }

  checkAndRetrain(channel: string = 'Choice', thresholdMape: number = 15): Observable<any> {
    return this.http.post(
      `${this.baseUrl}/check-and-retrain-lstm?channel=${channel}&threshold_mape=${thresholdMape}`,
      {}
    );
  }
}