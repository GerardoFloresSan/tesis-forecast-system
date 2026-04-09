import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { finalize } from 'rxjs';
import { SystemSummaryResponse } from '../../models/system-summary.model';
import { SystemSummaryService } from '../../services/system-summary.service';
import { ModelActionsService } from '../../services/model-actions.service';
import { ForecastActionsService } from '../../services/forecast-actions.service';
import { ChannelService } from '../../services/channel.service';
import { LimaDateTimePipe } from '../../shared/pipes/lima-datetime.pipe';

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, DecimalPipe, LimaDateTimePipe],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit, OnDestroy {
  private readonly systemSummaryService = inject(SystemSummaryService);
  private readonly modelActionsService = inject(ModelActionsService);
  private readonly forecastActionsService = inject(ForecastActionsService);
  private readonly channelService = inject(ChannelService);

  summary: SystemSummaryResponse | null = null;
  availableChannels: string[] = [];

  loading = false;
  errorMessage = '';
  channel = 'Choice';

  actionLoading = false;
  actionMessage = '';
  actionError = '';

  autoRefreshEnabled = true;
  autoRefreshIntervalSeconds = 30;
  lastAutoRefreshAt: Date | null = null;

  private autoRefreshTimerId: number | null = null;

  ngOnInit(): void {
    this.loadChannels();
    this.startAutoRefresh();
  }

  ngOnDestroy(): void {
    this.stopAutoRefresh();
  }

  loadChannels(): void {
    this.channelService.getChannels().subscribe({
      next: (channels) => {
        this.availableChannels = channels;

        if (channels.length > 0 && !channels.includes(this.channel)) {
          this.channel = channels[0];
        }

        this.loadSummary();
      },
      error: (error) => {
        console.error(error);
        this.availableChannels = ['Choice'];
        this.loadSummary();
      }
    });
  }

  loadSummary(silent: boolean = false): void {
    if (!silent) {
      this.loading = true;
      this.errorMessage = '';
    }

    this.systemSummaryService
      .getSummary(this.channel)
      .pipe(
        finalize(() => {
          if (!silent) {
            this.loading = false;
          }
        })
      )
      .subscribe({
        next: (response) => {
          this.summary = response;
          if (silent) {
            this.lastAutoRefreshAt = new Date();
          }
        },
        error: (error) => {
          console.error(error);
          if (!silent) {
            this.errorMessage = 'No se pudo cargar el resumen del sistema.';
          }
        }
      });
  }

  onChannelChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.channel = value;
    this.actionMessage = '';
    this.actionError = '';
    this.loadSummary();
  }

  trainLstm(): void {
    this.executeAction(
      () => this.modelActionsService.trainLstm(this.channel),
      'Entrenamiento LSTM ejecutado correctamente.'
    );
  }

  retrainLstm(): void {
    this.executeAction(
      () => this.modelActionsService.retrainLstm(this.channel),
      'Reentrenamiento LSTM ejecutado correctamente.'
    );
  }

  checkAndRetrain(): void {
    this.executeAction(
      () => this.modelActionsService.checkAndRetrain(this.channel, 15),
      'Check & Retrain ejecutado correctamente.'
    );
  }

  generateForecast(): void {
    this.executeAction(
      () => this.forecastActionsService.generateDailyForecast(this.channel),
      'Forecast manual generado correctamente.'
    );
  }

  toggleAutoRefresh(): void {
    this.autoRefreshEnabled = !this.autoRefreshEnabled;

    if (this.autoRefreshEnabled) {
      this.startAutoRefresh();
    } else {
      this.stopAutoRefresh();
    }
  }

  private startAutoRefresh(): void {
    this.stopAutoRefresh();

    if (!this.autoRefreshEnabled) {
      return;
    }

    this.autoRefreshTimerId = window.setInterval(() => {
      if (this.loading || this.actionLoading) {
        return;
      }

      this.loadSummary(true);
    }, this.autoRefreshIntervalSeconds * 1000);
  }

  private stopAutoRefresh(): void {
    if (this.autoRefreshTimerId !== null) {
      window.clearInterval(this.autoRefreshTimerId);
      this.autoRefreshTimerId = null;
    }
  }

  private executeAction(requestFactory: () => any, successFallbackMessage: string): void {
    this.actionLoading = true;
    this.actionMessage = '';
    this.actionError = '';

    requestFactory()
      .pipe(finalize(() => (this.actionLoading = false)))
      .subscribe({
        next: (response: any) => {
          const backendMessage =
            response?.message ||
            response?.detail ||
            successFallbackMessage;

          this.actionMessage = backendMessage;
          this.loadSummary();
        },
        error: (error: any) => {
          console.error(error);
          this.actionError =
            error?.error?.detail ||
            'Ocurrió un error al ejecutar la acción.';
        }
      });
  }

  getStatusClass(status: string | null | undefined): string {
    if (!status) return '';

    const normalized = status.toLowerCase();

    if (normalized === 'success' || normalized === 'activo') {
      return 'status-success';
    }

    if (normalized === 'failed' || normalized === 'error') {
      return 'status-failed';
    }

    if (normalized === 'running') {
      return 'status-running';
    }

    return '';
  }

  get modelActionsDisabled(): boolean {
    return this.channel !== 'Choice';
  }
}