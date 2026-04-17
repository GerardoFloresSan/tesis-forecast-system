import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { finalize, forkJoin } from 'rxjs';
import {
  ForecastHistoryItem,
  ForecastIntervalHistoryItem,
  LstmHistoryItem,
  SchedulerJobHistoryItem
} from '../../models/system-summary.model';
import { ForecastHistoryService } from '../../services/forecast-history.service';
import { ModelHistoryService } from '../../services/model-history.service';
import { SchedulerJobHistoryService } from '../../services/scheduler-job-history.service';
import { ChannelService } from '../../services/channel.service';
import { LimaDateTimePipe } from '../../shared/pipes/lima-datetime.pipe';

interface MapeBar {
  x: number;
  y: number;
  width: number;
  height: number;
  value: number;
  label: string;
  color: string;
  showLabel: boolean;
}

interface GridLine {
  y: number;
  label: string;
}

interface MapeChartData {
  bars: MapeBar[];
  hasData: boolean;
  thresholdY: number;
  gridLines: GridLine[];
  bottom: number;
}

@Component({
  selector: 'app-monitoring',
  imports: [CommonModule, DecimalPipe, LimaDateTimePipe],
  templateUrl: './monitoring.component.html',
  styleUrl: './monitoring.component.css'
})
export class MonitoringComponent implements OnInit {
  private readonly forecastHistoryService = inject(ForecastHistoryService);
  private readonly modelHistoryService = inject(ModelHistoryService);
  private readonly schedulerJobHistoryService = inject(SchedulerJobHistoryService);
  private readonly channelService = inject(ChannelService);

  availableChannels: string[] = [];
  channel = 'Choice';

  forecastHistory: ForecastHistoryItem[] = [];
  forecastIntervals: ForecastIntervalHistoryItem[] = [];
  modelHistory: LstmHistoryItem[] = [];
  schedulerHistory: SchedulerJobHistoryItem[] = [];

  loading = false;
  intervalLoading = false;
  errorMessage = '';
  intervalError = '';
  lastRefreshAt: Date | null = null;
  selectedForecastDate = '';
  selectedForecastRunId: number | null = null;

  ngOnInit(): void {
    this.loadChannels();
  }

  loadChannels(): void {
    this.channelService.getChannels().subscribe({
      next: (channels) => {
        this.availableChannels = channels;
        if (channels.length > 0 && !channels.includes(this.channel)) {
          this.channel = channels[0];
        }
        this.loadMonitoring();
      },
      error: (error) => {
        console.error(error);
        this.availableChannels = ['Choice', 'España'];
        this.loadMonitoring();
      }
    });
  }

  loadMonitoring(): void {
    this.loading = true;
    this.errorMessage = '';

    forkJoin({
      forecastHistory: this.forecastHistoryService.getHistory(this.channel, 20),
      modelHistory: this.modelHistoryService.getHistory(this.channel, 30),
      schedulerHistory: this.schedulerJobHistoryService.getHistory(30)
    })
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (response) => {
          this.forecastHistory = response.forecastHistory;
          this.modelHistory = response.modelHistory;
          this.schedulerHistory = response.schedulerHistory.filter(
            (item) => !item.channel || item.channel === this.channel
          );

          const availableDates = this.forecastHistory
            .map((item) => this.extractDateOnly(item.forecast_date))
            .filter((value): value is string => !!value);

          const preferredDate = this.selectedForecastDate && availableDates.includes(this.selectedForecastDate)
            ? this.selectedForecastDate
            : availableDates[0] ?? '';

          this.selectedForecastDate = preferredDate;
          this.selectedForecastRunId = this.forecastHistory.find(
            (item) => this.extractDateOnly(item.forecast_date) === preferredDate
          )?.id ?? null;

          if (preferredDate) {
            this.loadIntervalHistory(preferredDate);
          } else {
            this.forecastIntervals = [];
          }

          this.lastRefreshAt = new Date();
        },
        error: (error) => {
          console.error(error);
          this.errorMessage = 'No se pudo cargar la vista de monitoreo.';
        }
      });
  }

  loadIntervalHistory(forecastDate: string): void {
    this.intervalLoading = true;
    this.intervalError = '';

    this.forecastHistoryService
      .getIntervalHistory(this.channel, forecastDate, 2000)
      .pipe(finalize(() => (this.intervalLoading = false)))
      .subscribe({
        next: (items) => {
          this.forecastIntervals = items.sort((a, b) => a.slot_index - b.slot_index);
        },
        error: (error) => {
          console.error(error);
          this.forecastIntervals = [];
          this.intervalError = 'No se pudo cargar el detalle operativo por intervalos.';
        }
      });
  }

  onChannelChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.channel = value;
    this.selectedForecastDate = '';
    this.selectedForecastRunId = null;
    this.loadMonitoring();
  }

  selectForecastRun(item: ForecastHistoryItem): void {
    const forecastDate = this.extractDateOnly(item.forecast_date);
    if (!forecastDate) {
      return;
    }

    this.selectedForecastDate = forecastDate;
    this.selectedForecastRunId = item.id;
    this.loadIntervalHistory(forecastDate);
  }

  get selectedForecastDateLabel(): string {
    return this.formatDateOnly(this.selectedForecastDate);
  }

  get intervalDailyTotal(): number {
    return this.forecastIntervals.reduce((sum, item) => sum + item.predicted_value, 0);
  }

  get peakInterval(): ForecastIntervalHistoryItem | null {
    if (!this.forecastIntervals.length) {
      return null;
    }

    return this.forecastIntervals.reduce((peak, item) => (
      item.predicted_value > peak.predicted_value ? item : peak
    ));
  }

  getStatusClass(status: string | null | undefined): string {
    if (!status) return '';
    const normalized = status.toLowerCase();
    if (normalized === 'success') return 'status-success';
    if (normalized === 'failed' || normalized === 'error') return 'status-failed';
    if (normalized === 'running') return 'status-running';
    return '';
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

  get mapeChartData(): MapeChartData {
    const PAD_L = 48, PAD_R = 24, PAD_T = 14, PAD_B = 32;
    const W = 600, H = 160;
    const plotW = W - PAD_L - PAD_R;
    const plotH = H - PAD_T - PAD_B;
    const bottom = PAD_T + plotH;

    const empty: MapeChartData = { bars: [], hasData: false, thresholdY: 0, gridLines: [], bottom };

    const items = [...this.modelHistory]
      .filter((item) => item.mape != null)
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .slice(-15);

    if (items.length === 0) return empty;

    const mapes = items.map((item) => item.mape as number);
    const maxMape = Math.max(...mapes, 22);
    const minMape = 0;

    const toY = (value: number) =>
      PAD_T + (1 - (value - minMape) / (maxMape - minMape)) * plotH;

    const barSpacing = plotW / items.length;
    const barWidth = barSpacing * 0.55;

    const bars: MapeBar[] = items.map((item, index) => {
      const mapeValue = item.mape as number;
      const x = PAD_L + index * barSpacing + (barSpacing - barWidth) / 2;
      const y = toY(mapeValue);
      const height = bottom - y;

      let color: string;
      if (mapeValue < 10) color = '#15803d';
      else if (mapeValue < 15) color = '#2563eb';
      else if (mapeValue < 20) color = '#b45309';
      else color = '#b91c1c';

      const maxLabels = 8;
      const step = Math.max(1, Math.ceil(items.length / maxLabels));
      const showLabel = index % step === 0 || index === items.length - 1;

      return {
        x: +x.toFixed(1),
        y: +y.toFixed(1),
        width: +barWidth.toFixed(1),
        height: +height.toFixed(1),
        value: mapeValue,
        label: item.created_at.slice(5, 10),
        color,
        showLabel
      };
    });

    const thresholdY = +toY(15).toFixed(1);

    const gridLines: GridLine[] = [0, 10, 20]
      .filter((value) => value <= maxMape + 2)
      .map((value) => ({ y: +toY(value).toFixed(1), label: `${value}%` }));

    return { bars, hasData: true, thresholdY, gridLines, bottom };
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
