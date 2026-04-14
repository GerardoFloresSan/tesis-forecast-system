import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { finalize } from 'rxjs';
import { ForecastHistoryItem, SystemSummaryResponse } from '../../models/system-summary.model';
import { SystemSummaryService } from '../../services/system-summary.service';
import { ModelActionsService } from '../../services/model-actions.service';
import { ForecastActionsService } from '../../services/forecast-actions.service';
import { ChannelService } from '../../services/channel.service';
import { ForecastHistoryService } from '../../services/forecast-history.service';
import { LimaDateTimePipe } from '../../shared/pipes/lima-datetime.pipe';

// Tipos locales para los datos del gráfico SVG
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

  summary: SystemSummaryResponse | null = null;
  forecastHistory: ForecastHistoryItem[] = [];
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
        this.loadForecastHistory();
      },
      error: (error) => {
        console.error(error);
        this.availableChannels = ['Choice'];
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
      .pipe(finalize(() => { if (!silent) this.loading = false; }))
      .subscribe({
        next: (response) => {
          this.summary = response;
          if (silent) this.lastAutoRefreshAt = new Date();
        },
        error: (error) => {
          console.error(error);
          if (!silent) this.errorMessage = 'No se pudo cargar el resumen del sistema.';
        }
      });
  }

  // Carga el historial de forecasts para alimentar el gráfico de tendencia
  loadForecastHistory(silent: boolean = false): void {
    this.forecastHistoryService.getHistory(this.channel, 20).subscribe({
      next: (items) => { this.forecastHistory = items; },
      error: () => {} // Fallo silencioso: el gráfico no se mostrará
    });
  }

  onChannelChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.channel = value;
    this.actionMessage = '';
    this.actionError = '';
    this.loadSummary();
    this.loadForecastHistory();
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
    if (!this.autoRefreshEnabled) return;

    this.autoRefreshTimerId = window.setInterval(() => {
      if (this.loading || this.actionLoading) return;
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

  private executeAction(requestFactory: () => any, successFallbackMessage: string): void {
    this.actionLoading = true;
    this.actionMessage = '';
    this.actionError = '';

    requestFactory()
      .pipe(finalize(() => (this.actionLoading = false)))
      .subscribe({
        next: (response: any) => {
          this.actionMessage =
            response?.message || response?.detail || successFallbackMessage;
          this.loadSummary();
          this.loadForecastHistory();
        },
        error: (error: any) => {
          console.error(error);
          this.actionError =
            error?.error?.detail || 'Ocurrió un error al ejecutar la acción.';
        }
      });
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
    return this.channel !== 'Choice';
  }

  // ──────────────────────────────────────────────
  // Indicador de calidad del MAPE
  // ──────────────────────────────────────────────
  get mapeQuality(): MapeQuality {
    const mape = this.summary?.lstm_metrics?.mape;
    if (mape == null) {
      return { label: 'Sin datos', color: '#94a3b8', bgColor: '#f1f5f9', fillWidth: 0 };
    }
    // Barra que se llena hasta el 30 % como máximo visual
    const fillWidth = Math.min(mape, 30) / 30 * 100;
    if (mape < 10) return { label: 'Excelente', color: '#15803d', bgColor: '#dcfce7', fillWidth };
    if (mape < 15) return { label: 'Bueno',     color: '#2563eb', bgColor: '#dbeafe', fillWidth };
    if (mape < 20) return { label: 'Aceptable', color: '#b45309', bgColor: '#fef3c7', fillWidth };
    return            { label: 'Mejorable',  color: '#b91c1c', bgColor: '#fee2e2', fillWidth };
  }

  // ──────────────────────────────────────────────
  // Datos para el gráfico SVG de tendencia de pronósticos
  // viewBox="0 0 600 180"  PAD_L=52 PAD_R=16 PAD_T=14 PAD_B=32
  // plotW=532  plotH=134  bottom=148
  // ──────────────────────────────────────────────
  get forecastChartData(): ForecastChartData {
    const PAD_L = 52, PAD_R = 16, PAD_T = 14, PAD_B = 32;
    const W = 600, H = 180;
    const plotW = W - PAD_L - PAD_R;   // 532
    const plotH = H - PAD_T - PAD_B;   // 134
    const bottom = PAD_T + plotH;       // 148

    const empty: ForecastChartData = {
      linePath: '', areaPath: '', points: [], gridLines: [],
      xLabels: [], hasData: false, minLabel: '', maxLabel: ''
    };

    const items = [...this.forecastHistory]
      .filter(i => i.predicted_value != null)
      .sort((a, b) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime())
      .slice(-20);

    if (items.length < 2) return empty;

    const values = items.map(i => i.predicted_value);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const range  = rawMax - rawMin || rawMax * 0.1 || 1;
    const minY   = rawMin - range * 0.15;
    const maxY   = rawMax + range * 0.15;

    const toX = (i: number) => PAD_L + (i / (items.length - 1)) * plotW;
    const toY = (v: number) => PAD_T + (1 - (v - minY) / (maxY - minY)) * plotH;

    const points: ChartPoint[] = items.map((item, i) => ({
      x: +toX(i).toFixed(1),
      y: +toY(item.predicted_value).toFixed(1),
      value: item.predicted_value,
      label: item.forecast_date.slice(5, 10) // MM-DD
    }));

    const linePath = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`)
      .join(' ');

    const areaPath =
      `M${points[0].x},${bottom} ` +
      points.map(p => `L${p.x},${p.y}`).join(' ') +
      ` L${points[points.length - 1].x},${bottom} Z`;

    // 4 líneas de referencia horizontales (de arriba hacia abajo)
    const gridLines: GridLine[] = [0, 1, 2, 3].map(idx => {
      const t = 1 - idx / 3;
      const v = minY + t * (maxY - minY);
      return { y: +toY(v).toFixed(1), label: Math.round(v).toString() };
    });

    // Etiquetas del eje X: máximo 6 visibles
    const step = Math.max(1, Math.ceil(items.length / 6));
    const xLabels = points.filter((_, i) => i % step === 0 || i === points.length - 1);

    return {
      linePath, areaPath, points, gridLines, xLabels, hasData: true,
      minLabel: Math.round(rawMin).toString(),
      maxLabel: Math.round(rawMax).toString()
    };
  }
}
