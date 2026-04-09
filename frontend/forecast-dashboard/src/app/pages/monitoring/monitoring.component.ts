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
      modelHistory: this.modelHistoryService.getHistory(this.channel, 20),
      schedulerHistory: this.schedulerJobHistoryService.getHistory(30)
    })
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (response) => {
          this.forecastHistory = response.forecastHistory;
          this.modelHistory = response.modelHistory;
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

    if (normalized === 'success') {
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
}