import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';

interface LoginRequest {
  username: string;
  password: string;
}

interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  username: string;
  full_name: string;
  role: string;
}

interface CurrentUser {
  id: number;
  username: string;
  full_name: string;
  role: string;
  is_active: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly apiUrl = 'http://127.0.0.1:8000';
  private readonly tokenKey = 'forecast_token';
  private readonly userKey = 'forecast_user';

  currentUser = signal<CurrentUser | null>(this.getStoredUser());

  constructor(
    private http: HttpClient,
    private router: Router
  ) {}

  login(username: string, password: string): Observable<LoginResponse> {
    const body: LoginRequest = { username, password };

    return this.http.post<LoginResponse>(`${this.apiUrl}/auth/login`, body).pipe(
      tap((response) => {
        localStorage.setItem(this.tokenKey, response.access_token);

        const user: CurrentUser = {
          id: 0,
          username: response.username,
          full_name: response.full_name,
          role: response.role,
          is_active: true
        };

        localStorage.setItem(this.userKey, JSON.stringify(user));
        this.currentUser.set(user);
      })
    );
  }

  loadCurrentUser(): Observable<CurrentUser> {
    return this.http.get<CurrentUser>(`${this.apiUrl}/auth/me`).pipe(
      tap((user) => {
        localStorage.setItem(this.userKey, JSON.stringify(user));
        this.currentUser.set(user);
      })
    );
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
    this.currentUser.set(null);
    this.router.navigate(['/login']);
  }

  private getStoredUser(): CurrentUser | null {
    const rawUser = localStorage.getItem(this.userKey);

    if (!rawUser) {
      return null;
    }

    try {
      return JSON.parse(rawUser) as CurrentUser;
    } catch {
      localStorage.removeItem(this.userKey);
      return null;
    }
  }
}