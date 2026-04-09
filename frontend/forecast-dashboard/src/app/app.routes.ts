import { Routes } from '@angular/router';
import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { MonitoringComponent } from './pages/monitoring/monitoring.component';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: 'dashboard',
    component: DashboardComponent
  },
  {
    path: 'monitoring',
    component: MonitoringComponent
  },
  {
    path: '**',
    redirectTo: 'dashboard'
  }
];