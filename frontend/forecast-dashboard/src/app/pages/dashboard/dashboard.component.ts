import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { finalize } from 'rxjs';
import {
  ForecastBatchResponse,
  ForecastHistoryItem,
  ForecastIntervalHistoryItem,
  SystemSummaryResponse
} from '../../models/system-summary.model';
import { SystemSummaryService } from '../../services/system-summary.service';
import { ModelActionsService } from '../../services/model-actions.service';
import { ForecastActionsService } from '../../services/forecast-actions.service';
import { ChannelService } from '../../services/channel.service';
import { ForecastHistoryService } from '../../services/forecast-history.service';
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
  imports: [CommonModule, DecimalPipe, LimaDateTimePipe],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit, OnDestroy {
  private readonly systemSummaryService = inject(SystemSummaryService);
  private readonly modelActionsService = inject(ModelActionsService);
  private readonly forecastActionsService = inject(ForecastActionsService);
  private readonly channelService = inject(ChannelService);
  private readonly forecastHistoryService = inject(ForecastHistoryService);

  private readonly forecastEnabledChannels = ['choice', 'espana'];

  summary: SystemSummaryResponse | null = null;
  forecastHistory: ForecastHistoryItem[] = [];
  forecastIntervals: ForecastIntervalHistoryItem[] = [];
  availableChannels: string[] = [];

  loading = false;
  errorMessage = '';
  channel = 'Choice';

  actionLoading = false;
  actionMessage = '';
  actionError = '';

  intervalLoading = false;
  intervalError = '';
  selectedForecastDate = '';
  selectedForecastRunId: number | null = null;

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
      },
      error: (error) => {
        console.error(error);
        this.availableChannels = ['Choice', 'España'];
        this.loadSummary();
        this.loadForecastHistory();
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
          this.intervalError = '';
          return;
        }

        this.selectedForecastDate = preferredDate;
        const selectedRun = items.find((item) => this.extractDateOnly(item.forecast_date) === preferredDate);
        this.selectedForecastRunId = selectedRun?.id ?? null;
        this.loadForecastIntervals(preferredDate, silent);
      },
      error: (error) => {
        console.error(error);
        this.forecastHistory = [];
        this.forecastIntervals = [];
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
    this.selectedForecastDate = '';
    this.selectedForecastRunId = null;
    this.loadSummary();
    this.loadForecastHistory();
  }

  selectForecastRun(item: ForecastHistoryItem): void {
    const forecastDate = this.extractDateOnly(item.forecast_date);
    if (!forecastDate) {
      return;
    }

    this.selectedForecastRunId = item.id;
    this.selectedForecastDate = forecastDate;
    this.loadForecastIntervals(forecastDate);
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
      return { label: 'Sin datos', color: '#94a3b8', bgColor: '#f1f5f9', fillWidth: 0 };
    }

    const fillWidth = Math.min(mape, 30) / 30 * 100;
    if (mape < 10) return { label: 'Excelente', color: '#15803d', bgColor: '#dcfce7', fillWidth };
    if (mape < 15) return { label: 'Bueno', color: '#2563eb', bgColor: '#dbeafe', fillWidth };
    if (mape < 20) return { label: 'Aceptable', color: '#b45309', bgColor: '#fef3c7', fillWidth };
    return { label: 'Mejorable', color: '#b91c1c', bgColor: '#fee2e2', fillWidth };
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

  private startAutoRefresh(): void {
    this.stopAutoRefresh();
    if (!this.autoRefreshEnabled) return;

    this.autoRefreshTimerId = window.setInterval(() => {
      if (this.loading || this.actionLoading || this.intervalLoading) return;
      this.loadSummary(true);
      this.loadForecastHistory(true);
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
}
