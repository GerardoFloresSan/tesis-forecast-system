import { Routes } from '@angular/router';

import { LoginComponent } from './pages/login/login.component';
import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { MonitoringComponent } from './pages/monitoring/monitoring.component';
import { UploadComponent } from './pages/upload/upload.component';
import { ExternalVariablesComponent } from './pages/external-variables/external-variables.component';
import { QualityComponent } from './pages/quality/quality.component';
import { authGuard } from './core/auth/auth.guard';

export const routes: Routes = [
  {
    path: 'login',
    component: LoginComponent
  },
  {
    path: 'dashboard',
    component: DashboardComponent,
    canActivate: [authGuard]
  },
  {
    path: 'monitoring',
    component: MonitoringComponent,
    canActivate: [authGuard]
  },
  {
    path: 'upload',
    component: UploadComponent,
    canActivate: [authGuard]
  },
  {
    path: 'external-variables',
    component: ExternalVariablesComponent,
    canActivate: [authGuard]
  },
  {
    path: 'quality',
    component: QualityComponent,
    canActivate: [authGuard]
  },
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: '**',
    redirectTo: 'dashboard'
  }
];
