import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { finalize, forkJoin } from 'rxjs';
import {
  ForecastBatchResponse,
  ForecastHistoryItem,
  ForecastIntervalHistoryItem,
  ForecastMonitoringResponse,
  SLAAlertResponse,
  SystemSummaryResponse
} from '../../models/system-summary.model';
import { SystemSummaryService } from '../../services/system-summary.service';
import { ModelActionsService } from '../../services/model-actions.service';
import { ForecastActionsService } from '../../services/forecast-actions.service';
import { ChannelService } from '../../services/channel.service';
import { ForecastHistoryService } from '../../services/forecast-history.service';
import { ForecastMonitoringService } from '../../services/forecast-monitoring.service';
import { AlertsService } from '../../services/alerts.service';
import { LimaDateTimePipe } from '../../shared/pipes/lima-datetime.pipe';

interface ChartPoint {
  x: number;
  y: number;
  value: number;
  label: string;
}

interface GridLine {
  y: number;
  label: string;
}

interface ForecastChartData {
  linePath: string;
  areaPath: string;
  points: ChartPoint[];
  gridLines: GridLine[];
  xLabels: ChartPoint[];
  hasData: boolean;
  minLabel: string;
  maxLabel: string;
}

interface IntervalChartBar {
  x: number;
  y: number;
  width: number;
  height: number;
  value: number;
  label: string;
  showLabel: boolean;
}

interface IntervalChartData {
  bars: IntervalChartBar[];
  gridLines: GridLine[];
  hasData: boolean;
  bottom: number;
  maxLabel: string;
}

interface MapeQuality {
  label: string;
  color: string;
  bgColor: string;
  fillWidth: number;
}

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, DecimalPipe, LimaDateTimePipe, RouterLink],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit, OnDestroy {
  private readonly systemSummaryService = inject(SystemSummaryService);
  private readonly modelActionsService = inject(ModelActionsService);
  private readonly forecastActionsService = inject(ForecastActionsService);
  private readonly channelService = inject(ChannelService);
  private readonly forecastHistoryService = inject(ForecastHistoryService);
  private readonly forecastMonitoringService = inject(ForecastMonitoringService);
  private readonly alertsService = inject(AlertsService);
  private readonly forecastEnabledChannels = ['choice', 'espana'];

  summary: SystemSummaryResponse | null = null;
  forecastHistory: ForecastHistoryItem[] = [];
  forecastIntervals: ForecastIntervalHistoryItem[] = [];
  availableChannels: string[] = [];

  monitoringSummary: ForecastMonitoringResponse | null = null;
  monitoringByDate: ForecastMonitoringResponse | null = null;
  activeAlerts: SLAAlertResponse[] = [];
  alertHistory: SLAAlertResponse[] = [];

  loading = false;
  errorMessage = '';
  channel = 'Choice';

  actionLoading = false;
  actionMessage = '';
  actionError = '';

  intervalLoading = false;
  intervalError = '';

  monitoringLoading = false;
  alertsLoading = false;
  monitoringError = '';
  alertError = '';
  selectedForecastDate = '';
  selectedForecastRunId: number | null = null;

  readonly pageSize = 10;
  alertHistoryPage = 1;
  forecastHistoryPage = 1;
  forecastIntervalsPage = 1;

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
        this.loadForecastHistory();
        this.loadMonitoringSummary();
        this.loadAlerts();
      },
      error: (error) => {
        console.error(error);
        this.availableChannels = ['Choice', 'España'];
        this.loadSummary();
        this.loadForecastHistory();
        this.loadMonitoringSummary();
        this.loadAlerts();
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
      .pipe(finalize(() => {
        if (!silent) {
          this.loading = false;
        }
      }))
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

  loadForecastHistory(silent: boolean = false): void {
    this.forecastHistoryService.getHistory(this.channel, 20).subscribe({
      next: (items) => {
        this.forecastHistory = items;
        this.forecastHistoryPage = 1;

        const availableDates = items
          .map((item) => this.extractDateOnly(item.forecast_date))
          .filter((value): value is string => !!value);

        const preferredDate = this.selectedForecastDate && availableDates.includes(this.selectedForecastDate)
          ? this.selectedForecastDate
          : availableDates[0] ?? '';

        if (!preferredDate) {
          this.selectedForecastDate = '';
          this.selectedForecastRunId = null;
          this.forecastIntervals = [];
          this.monitoringByDate = null;
          this.intervalError = '';
          return;
        }

        this.selectedForecastDate = preferredDate;
        const selectedRun = items.find((item) => this.extractDateOnly(item.forecast_date) === preferredDate);
        this.selectedForecastRunId = selectedRun?.id ?? null;
        this.loadForecastIntervals(preferredDate, silent);
        this.loadMonitoringByDate(preferredDate, silent);
      },
      error: (error) => {
        console.error(error);
        this.forecastHistory = [];
        this.forecastIntervals = [];
        this.monitoringByDate = null;
        this.selectedForecastDate = '';
        this.selectedForecastRunId = null;
      }
    });
  }

  loadForecastIntervals(forecastDate: string, silent: boolean = false): void {
    if (!silent) {
      this.intervalLoading = true;
      this.intervalError = '';
    }

    this.forecastHistoryService
      .getIntervalHistory(this.channel, forecastDate, 2000)
      .pipe(finalize(() => {
        if (!silent) {
          this.intervalLoading = false;
        }
      }))
      .subscribe({
        next: (items) => {
          this.forecastIntervals = items.sort((a, b) => a.slot_index - b.slot_index);
          this.forecastIntervalsPage = 1;
          if (silent) {
            this.lastAutoRefreshAt = new Date();
          }
        },
        error: (error) => {
          console.error(error);
          this.forecastIntervals = [];
          if (!silent) {
            this.intervalError = 'No se pudo cargar el detalle por intervalos.';
          }
        }
      });
  }

  onChannelChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.channel = value;
    this.actionMessage = '';
    this.actionError = '';
    this.intervalError = '';
    this.monitoringError = '';
    this.alertError = '';
    this.selectedForecastDate = '';
    this.selectedForecastRunId = null;
    this.monitoringByDate = null;
    this.loadSummary();
    this.loadForecastHistory();
    this.loadMonitoringSummary();
    this.loadAlerts();
  }

  selectForecastRun(item: ForecastHistoryItem): void {
    const forecastDate = this.extractDateOnly(item.forecast_date);
    if (!forecastDate) {
      return;
    }

    this.selectedForecastRunId = item.id;
    this.selectedForecastDate = forecastDate;
    this.loadForecastIntervals(forecastDate);
    this.loadMonitoringByDate(forecastDate);
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
      () => this.modelActionsService.checkAndRetrain(this.channel, 20),
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

  getStatusClass(status: string | null | undefined): string {
    if (!status) return '';
    const normalized = status.toLowerCase();
    if (normalized === 'success' || normalized === 'activo') return 'status-success';
    if (normalized === 'failed' || normalized === 'error') return 'status-failed';
    if (normalized === 'running') return 'status-running';
    return '';
  }

  get modelActionsDisabled(): boolean {
    return !this.isForecastChannelEnabled(this.channel);
  }

  get forecastActionsDisabled(): boolean {
    return !this.isForecastChannelEnabled(this.channel);
  }

  get selectedForecastHeader(): ForecastHistoryItem | null {
    if (this.selectedForecastRunId == null) {
      return this.forecastHistory[0] ?? null;
    }

    return this.forecastHistory.find((item) => item.id === this.selectedForecastRunId) ?? this.forecastHistory[0] ?? null;
  }

  get pagedAlertHistory(): SLAAlertResponse[] { return this.paginate(this.alertHistory, this.alertHistoryPage); }
  get alertHistoryTotalPages(): number { return this.getTotalPages(this.alertHistory.length); }
  get pagedForecastHistory(): ForecastHistoryItem[] { return this.paginate(this.forecastHistory, this.forecastHistoryPage); }
  get forecastHistoryTotalPages(): number { return this.getTotalPages(this.forecastHistory.length); }
  get pagedForecastIntervals(): ForecastIntervalHistoryItem[] { return this.paginate(this.forecastIntervals, this.forecastIntervalsPage); }
  get forecastIntervalsTotalPages(): number { return this.getTotalPages(this.forecastIntervals.length); }

  goToAlertHistoryPage(page: number): void { this.alertHistoryPage = this.clampPage(page, this.alertHistoryTotalPages); }
  goToForecastHistoryPage(page: number): void { this.forecastHistoryPage = this.clampPage(page, this.forecastHistoryTotalPages); }
  goToForecastIntervalsPage(page: number): void { this.forecastIntervalsPage = this.clampPage(page, this.forecastIntervalsTotalPages); }

  get selectedForecastDateLabel(): string {
    return this.formatDateOnly(this.selectedForecastDate);
  }

  get intervalDailyTotal(): number {
    return this.forecastIntervals.reduce((sum, item) => sum + item.predicted_value, 0);
  }

  get intervalAverage(): number {
    if (!this.forecastIntervals.length) {
      return 0;
    }

    return this.intervalDailyTotal / this.forecastIntervals.length;
  }

  get peakInterval(): ForecastIntervalHistoryItem | null {
    if (!this.forecastIntervals.length) {
      return null;
    }

    return this.forecastIntervals.reduce((peak, item) => (
      item.predicted_value > peak.predicted_value ? item : peak
    ));
  }

  get latestOperationalModelVersion(): string {
    return this.forecastIntervals[0]?.model_version
      || this.selectedForecastHeader?.model_version
      || this.summary?.latest_forecast?.model_version
      || '-';
  }

  get intervalChartData(): IntervalChartData {
    const PAD_L = 52, PAD_R = 18, PAD_T = 18, PAD_B = 34;
    const W = 720, H = 240;
    const plotW = W - PAD_L - PAD_R;
    const plotH = H - PAD_T - PAD_B;
    const bottom = PAD_T + plotH;

    const empty: IntervalChartData = {
      bars: [],
      gridLines: [],
      hasData: false,
      bottom,
      maxLabel: ''
    };

    if (!this.forecastIntervals.length) {
      return empty;
    }

    const values = this.forecastIntervals.map((item) => item.predicted_value);
    const maxValue = Math.max(...values, 1);
    const toY = (value: number) => PAD_T + (1 - (value / maxValue)) * plotH;

    const spacing = plotW / this.forecastIntervals.length;
    const barWidth = Math.max(8, spacing * 0.68);
    const labelStep = Math.max(1, Math.ceil(this.forecastIntervals.length / 8));

    const bars: IntervalChartBar[] = this.forecastIntervals.map((item, index) => {
      const x = PAD_L + index * spacing + (spacing - barWidth) / 2;
      const y = toY(item.predicted_value);
      const height = bottom - y;

      return {
        x: +x.toFixed(1),
        y: +y.toFixed(1),
        width: +barWidth.toFixed(1),
        height: +height.toFixed(1),
        value: item.predicted_value,
        label: item.interval_time.slice(0, 5),
        showLabel: index % labelStep === 0 || index === this.forecastIntervals.length - 1
      };
    });

    const gridLines: GridLine[] = [0, 1, 2, 3].map((idx) => {
      const value = (maxValue / 3) * idx;
      return {
        y: +toY(value).toFixed(1),
        label: Math.round(value).toString()
      };
    }).reverse();

    return {
      bars,
      gridLines,
      hasData: true,
      bottom,
      maxLabel: Math.round(maxValue).toString()
    };
  }

  get mapeQuality(): MapeQuality {
    const mape = this.summary?.lstm_metrics?.mape;
    if (mape == null) {
      return { label: 'Sin datos', color: '#9ca3af', bgColor: '#f3f4f6', fillWidth: 0 };
    }

    const fillWidth = Math.min(mape, 30) / 30 * 100;
    if (mape <= 15) return { label: 'Óptimo', color: '#16a34a', bgColor: 'rgba(22, 163, 74, 0.08)', fillWidth };
    if (mape <= 20) return { label: 'Aceptable', color: '#ca8a04', bgColor: 'rgba(202, 138, 4, 0.08)', fillWidth };
    return { label: 'Deficiente', color: '#dc2626', bgColor: 'rgba(220, 38, 38, 0.06)', fillWidth };
  }

  get forecastChartData(): ForecastChartData {
    const PAD_L = 52, PAD_R = 16, PAD_T = 14, PAD_B = 32;
    const W = 600, H = 180;
    const plotW = W - PAD_L - PAD_R;
    const plotH = H - PAD_T - PAD_B;
    const bottom = PAD_T + plotH;

    const empty: ForecastChartData = {
      linePath: '',
      areaPath: '',
      points: [],
      gridLines: [],
      xLabels: [],
      hasData: false,
      minLabel: '',
      maxLabel: ''
    };

    const items = [...this.forecastHistory]
      .filter((item) => item.predicted_value != null)
      .sort((a, b) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime())
      .slice(-20);

    if (items.length < 2) {
      return empty;
    }

    const values = items.map((item) => item.predicted_value);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const range = rawMax - rawMin || rawMax * 0.1 || 1;
    const minY = rawMin - range * 0.15;
    const maxY = rawMax + range * 0.15;

    const toX = (index: number) => PAD_L + (index / (items.length - 1)) * plotW;
    const toY = (value: number) => PAD_T + (1 - (value - minY) / (maxY - minY)) * plotH;

    const points: ChartPoint[] = items.map((item, index) => ({
      x: +toX(index).toFixed(1),
      y: +toY(item.predicted_value).toFixed(1),
      value: item.predicted_value,
      label: this.extractDateOnly(item.forecast_date)?.slice(5, 10) ?? item.forecast_date.slice(5, 10)
    }));

    const linePath = points
      .map((point, index) => `${index === 0 ? 'M' : 'L'}${point.x},${point.y}`)
      .join(' ');

    const areaPath =
      `M${points[0].x},${bottom} ` +
      points.map((point) => `L${point.x},${point.y}`).join(' ') +
      ` L${points[points.length - 1].x},${bottom} Z`;

    const gridLines: GridLine[] = [0, 1, 2, 3].map((idx) => {
      const t = 1 - idx / 3;
      const value = minY + t * (maxY - minY);
      return { y: +toY(value).toFixed(1), label: Math.round(value).toString() };
    });

    const step = Math.max(1, Math.ceil(items.length / 6));
    const xLabels = points.filter((_, index) => index % step === 0 || index === points.length - 1);

    return {
      linePath,
      areaPath,
      points,
      gridLines,
      xLabels,
      hasData: true,
      minLabel: Math.round(rawMin).toString(),
      maxLabel: Math.round(rawMax).toString()
    };
  }

  getShiftLabel(label: string | null | undefined): string {
    const value = (label || '').toLowerCase();
    if (value === 'morning') return 'Mañana';
    if (value === 'afternoon') return 'Tarde';
    return label || '-';
  }

  isSelectedForecastRun(item: ForecastHistoryItem): boolean {
    return item.id === this.selectedForecastRunId;
  }

  loadMonitoringSummary(silent: boolean = false): void {
    if (!silent) {
      this.monitoringLoading = true;
      this.monitoringError = '';
    }

    this.forecastMonitoringService
      .getSummary(this.channel)
      .pipe(finalize(() => {
        if (!silent) {
          this.monitoringLoading = false;
        }
      }))
      .subscribe({
        next: (response) => {
          this.monitoringSummary = response;
        },
        error: (error) => {
          console.error(error);
          this.monitoringSummary = null;
          if (!silent) {
            this.monitoringError = 'No se pudo cargar el resumen de monitoreo.';
          }
        }
      });
  }

  loadMonitoringByDate(forecastDate: string, silent: boolean = false): void {
    if (!forecastDate) {
      this.monitoringByDate = null;
      return;
    }

    if (!silent) {
      this.monitoringLoading = true;
      this.monitoringError = '';
    }

    this.forecastMonitoringService
      .getByDate(this.channel, forecastDate)
      .pipe(finalize(() => {
        if (!silent) {
          this.monitoringLoading = false;
        }
      }))
      .subscribe({
        next: (response) => {
          this.monitoringByDate = response;
        },
        error: (error) => {
          console.error(error);
          this.monitoringByDate = null;
          if (!silent) {
            this.monitoringError = 'No se pudo cargar la comparación forecast vs real.';
          }
        }
      });
  }

  loadAlerts(silent: boolean = false): void {
    if (!silent) {
      this.alertsLoading = true;
      this.alertError = '';
    }

    forkJoin({
      active: this.alertsService.getActive(this.channel, 10),
      history: this.alertsService.getHistory(this.channel, 20)
    })
      .pipe(finalize(() => {
        if (!silent) {
          this.alertsLoading = false;
        }
      }))
      .subscribe({
        next: (response) => {
          this.activeAlerts = response.active;
          this.alertHistory = response.history;
        },
        error: (error) => {
          console.error(error);
          this.activeAlerts = [];
          this.alertHistory = [];
          if (!silent) {
            this.alertError = 'No se pudieron cargar las alertas del canal.';
          }
        }
      });
  }

  printDashboard(): void {
    window.print();
  }

  get selectedMonitoring(): ForecastMonitoringResponse | null {
    return this.monitoringByDate ?? this.monitoringSummary;
  }

  get activeAlertCount(): number {
    return this.activeAlerts.length;
  }

  get activeCriticalAlertsCount(): number {
    return this.activeAlerts.filter((item) => item.risk_level === 'critical').length;
  }

  get selectedMonitoringRiskLabel(): string {
    const value = (this.selectedMonitoring?.risk_level || '').toLowerCase();
    if (value === 'critical') return 'Crítico';
    if (value === 'warning') return 'Advertencia';
    if (value === 'normal') return 'Normal';
    if (value === 'unknown') return 'Sin datos reales';
    return this.selectedMonitoring?.risk_level || '-';
  }

  get selectedMonitoringErrorLabel(): string {
    const value = (this.selectedMonitoring?.error_level || '').toLowerCase();
    if (value === 'high_error') return 'Error alto';
    if (value === 'medium_error') return 'Error medio';
    if (value === 'low_error') return 'Error bajo';
    if (value === 'no_actual_data') return 'Sin data real';
    return this.selectedMonitoring?.error_level || '-';
  }

  getRiskClass(riskLevel: string | null | undefined): string {
    const value = (riskLevel || '').toLowerCase();
    if (value === 'critical') return 'risk-critical';
    if (value === 'warning') return 'risk-warning';
    if (value === 'normal') return 'risk-normal';
    return 'risk-unknown';
  }

  getAlertStatusLabel(status: string | null | undefined): string {
    const value = (status || '').toLowerCase();
    if (value === 'active') return 'Activa';
    if (value === 'acknowledged') return 'Reconocida';
    if (value === 'resolved') return 'Resuelta';
    return status || '-';
  }

  private paginate<T>(items: T[], page: number): T[] {
    const start = (page - 1) * this.pageSize;
    return items.slice(start, start + this.pageSize);
  }

  private getTotalPages(totalItems: number): number {
    return Math.max(1, Math.ceil(totalItems / this.pageSize));
  }

  private clampPage(page: number, totalPages: number): number {
    return Math.min(Math.max(page, 1), totalPages);
  }

  private startAutoRefresh(): void {
    this.stopAutoRefresh();
    if (!this.autoRefreshEnabled) return;

    this.autoRefreshTimerId = window.setInterval(() => {
      if (this.loading || this.actionLoading || this.intervalLoading) return;
      this.loadSummary(true);
      this.loadForecastHistory(true);
      this.loadMonitoringSummary(true);
      this.loadAlerts(true);
    }, this.autoRefreshIntervalSeconds * 1000);
  }

  private stopAutoRefresh(): void {
    if (this.autoRefreshTimerId !== null) {
      window.clearInterval(this.autoRefreshTimerId);
      this.autoRefreshTimerId = null;
    }
  }

  private executeAction(
    requestFactory: () => ReturnType<ForecastActionsService['generateDailyForecast']> | any,
    successFallbackMessage: string
  ): void {
    this.actionLoading = true;
    this.actionMessage = '';
    this.actionError = '';

    requestFactory()
      .pipe(finalize(() => (this.actionLoading = false)))
      .subscribe({
        next: (response: ForecastBatchResponse | any) => {
          this.actionMessage = response?.message || response?.detail || successFallbackMessage;

          if (response?.forecast_date) {
            this.selectedForecastDate = this.extractDateOnly(response.forecast_date) ?? this.selectedForecastDate;
            this.selectedForecastRunId = response?.id ?? this.selectedForecastRunId;
          }

          this.loadSummary();
          this.loadForecastHistory();
          this.loadMonitoringSummary();
          this.loadAlerts();
        },
        error: (error: any) => {
          console.error(error);
          this.actionError = error?.error?.detail || 'Ocurrió un error al ejecutar la acción.';
        }
      });
  }

  private isForecastChannelEnabled(channel: string): boolean {
    return this.forecastEnabledChannels.includes(this.normalizeChannel(channel));
  }

  private normalizeChannel(channel: string | null | undefined): string {
    return (channel || '')
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .trim()
      .toLowerCase();
  }

  private extractDateOnly(value: string | null | undefined): string | null {
    if (!value) {
      return null;
    }

    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
      return value;
    }

    const match = value.match(/^(\d{4}-\d{2}-\d{2})/);
    return match ? match[1] : null;
  }

  private formatDateOnly(value: string | null | undefined): string {
    if (!value) {
      return '-';
    }

    const parts = value.split('-');
    if (parts.length !== 3) {
      return value;
    }

    return `${parts[2]}/${parts[1]}/${parts[0]}`;
  }


  calculateRequiredAgents(
    forecast: number | string | null | undefined,
    aht: number | string | null | undefined
  ): number {
    const forecastValue = Number(forecast ?? 0);
    const ahtValue = Number(aht ?? 0);

    if (!forecastValue || forecastValue <= 0 || !ahtValue || ahtValue <= 0) {
      return 0;
    }

    const slotDurationSeconds = 1800; // 30 minutos
    const concurrency = 4; // chat concurrente

    const workloadSeconds = forecastValue * ahtValue;
    const requiredAgents = workloadSeconds / slotDurationSeconds / concurrency;

    return Math.ceil(requiredAgents);
  }

}
