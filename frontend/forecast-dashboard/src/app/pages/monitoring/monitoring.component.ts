import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { finalize, forkJoin } from 'rxjs';
import {
  ForecastHistoryItem,
  LstmHistoryItem,
  SchedulerJobHistoryItem
} from '../../models/system-summary.model';
import { ForecastHistoryService } from '../../services/forecast-history.service';
import { ModelHistoryService } from '../../services/model-history.service';
import { SchedulerJobHistoryService } from '../../services/scheduler-job-history.service';
import { ChannelService } from '../../services/channel.service';
import { LimaDateTimePipe } from '../../shared/pipes/lima-datetime.pipe';

// Tipos locales para el gráfico de MAPE
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
  modelHistory: LstmHistoryItem[] = [];
  schedulerHistory: SchedulerJobHistoryItem[] = [];

  loading = false;
  errorMessage = '';
  lastRefreshAt: Date | null = null;

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
        this.availableChannels = ['Choice'];
        this.loadMonitoring();
      }
    });
  }

  loadMonitoring(): void {
    this.loading = true;
    this.errorMessage = '';

    forkJoin({
      forecastHistory: this.forecastHistoryService.getHistory(this.channel, 20),
      modelHistory:    this.modelHistoryService.getHistory(this.channel, 30),
      schedulerHistory: this.schedulerJobHistoryService.getHistory(30)
    })
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (response) => {
          this.forecastHistory = response.forecastHistory;
          this.modelHistory    = response.modelHistory;
          this.schedulerHistory = response.schedulerHistory.filter(
            item => !item.channel || item.channel === this.channel
          );
          this.lastRefreshAt = new Date();
        },
        error: (error) => {
          console.error(error);
          this.errorMessage = 'No se pudo cargar la vista de monitoreo.';
        }
      });
  }

  onChannelChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.channel = value;
    this.loadMonitoring();
  }

  getStatusClass(status: string | null | undefined): string {
    if (!status) return '';
    const normalized = status.toLowerCase();
    if (normalized === 'success') return 'status-success';
    if (normalized === 'failed' || normalized === 'error') return 'status-failed';
    if (normalized === 'running') return 'status-running';
    return '';
  }

  // ──────────────────────────────────────────────────────────
  // Gráfico de barras MAPE por corrida del modelo (SVG nativo)
  // viewBox="0 0 600 160"  PAD_L=48 PAD_R=24 PAD_T=14 PAD_B=32
  // plotW=528  plotH=114  bottom=128
  // ──────────────────────────────────────────────────────────
  get mapeChartData(): MapeChartData {
    const PAD_L = 48, PAD_R = 24, PAD_T = 14, PAD_B = 32;
    const W = 600, H = 160;
    const plotW = W - PAD_L - PAD_R;  // 528
    const plotH = H - PAD_T - PAD_B;  // 114
    const bottom = PAD_T + plotH;      // 128

    const empty: MapeChartData = { bars: [], hasData: false, thresholdY: 0, gridLines: [], bottom };

    // Corridas con MAPE registrado, ordenadas cronológicamente, últimas 15
    const items = [...this.modelHistory]
      .filter(i => i.mape != null)
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .slice(-15);

    if (items.length === 0) return empty;

    const mapes  = items.map(i => i.mape as number);
    const maxMape = Math.max(...mapes, 22); // mínimo 22 % de techo visual
    const minMape = 0;

    const toY = (v: number) =>
      PAD_T + (1 - (v - minMape) / (maxMape - minMape)) * plotH;

    const barSpacing = plotW / items.length;
    const barW = barSpacing * 0.55;

    const bars: MapeBar[] = items.map((item, i) => {
      const mapeVal = item.mape as number;
      const x   = PAD_L + i * barSpacing + (barSpacing - barW) / 2;
      const y   = toY(mapeVal);
      const barH = bottom - y;

      let color: string;
      if      (mapeVal < 10) color = '#15803d';
      else if (mapeVal < 15) color = '#2563eb';
      else if (mapeVal < 20) color = '#b45309';
      else                   color = '#b91c1c';

      // Mostrar etiqueta de fecha en máximo 8 barras visibles
      const maxLabels = 8;
      const step = Math.max(1, Math.ceil(items.length / maxLabels));
      const showLabel = i % step === 0 || i === items.length - 1;

      return {
        x: +x.toFixed(1), y: +y.toFixed(1),
        width: +barW.toFixed(1), height: +barH.toFixed(1),
        value: mapeVal,
        label: item.created_at.slice(5, 10), // MM-DD
        color, showLabel
      };
    });

    const thresholdY = +toY(15).toFixed(1);

    const gridLines: GridLine[] = [0, 10, 20]
      .filter(v => v <= maxMape + 2)
      .map(v => ({ y: +toY(v).toFixed(1), label: `${v}%` }));

    return { bars, hasData: true, thresholdY, gridLines, bottom };
  }
}
