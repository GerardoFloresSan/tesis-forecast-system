import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  ForecastIntervalHistoryItem,
  ForecastMonthlyResponse,
  ForecastMonthlyStatusResponse
} from '../../models/system-summary.model';
import { ChannelService } from '../../services/channel.service';
import { ForecastActionsService } from '../../services/forecast-actions.service';
import { ForecastHistoryService } from '../../services/forecast-history.service';

@Component({
  selector: 'app-forecast-monthly-intervals',
  standalone: true,
  imports: [CommonModule, FormsModule, DatePipe, DecimalPipe],
  templateUrl: './forecast-monthly-intervals.component.html',
  styleUrl: './forecast-monthly-intervals.component.css'
})
export class ForecastMonthlyIntervalsComponent implements OnInit {
  private readonly forecastHistoryService = inject(ForecastHistoryService);
  private readonly forecastActionsService = inject(ForecastActionsService);
  private readonly channelService = inject(ChannelService);

  channels: string[] = ['Choice', 'España'];
  selectedChannel = 'Choice';

  startDate = '2026-05-01';
  endDate = '2026-05-31';

  forecastIntervals: ForecastIntervalHistoryItem[] = [];
  pagedForecastIntervals: ForecastIntervalHistoryItem[] = [];

  currentPage = 1;
  pageSize = 10;
  totalPages = 0;
  totalRecords = 0;

  loading = false;
  generating = false;
  errorMessage = '';
  successMessage = '';
  monthlyStatus: ForecastMonthlyStatusResponse | null = null;
  checkingStatus = false;

  totalForecast = 0;
  averageAht = 0;
  totalRequiredAgents = 0;

  ngOnInit(): void {
    this.loadChannels();
    this.refreshMonthlyStatus();
    this.loadMonthlyForecast();
  }

  loadChannels(): void {
    this.channelService.getChannels().subscribe({
      next: (channels) => {
        if (channels.length > 0) {
          this.channels = channels;
          if (!this.channels.includes(this.selectedChannel)) {
            this.selectedChannel = this.channels[0];
          }
        }
      },
      error: () => {
        this.channels = ['Choice', 'España'];
      }
    });
  }


  refreshMonthlyStatus(): void {
    this.checkingStatus = true;

    this.forecastActionsService
      .getMonthlyForecastStatus(this.selectedChannel, this.startDate, this.endDate)
      .subscribe({
        next: (response) => {
          this.monthlyStatus = response;
          this.checkingStatus = false;
        },
        error: () => {
          this.monthlyStatus = null;
          this.checkingStatus = false;
        }
      });
  }

  loadMonthlyForecast(): void {
    this.loading = true;
    this.errorMessage = '';
    this.successMessage = '';

    this.forecastHistoryService
      .getIntervalHistoryByRange(this.selectedChannel, this.startDate, this.endDate, 10000)
      .subscribe({
        next: (data) => {
          this.forecastIntervals = data;
          this.totalRecords = data.length;
          this.totalPages = Math.ceil(this.totalRecords / this.pageSize);
          this.currentPage = 1;
          this.calculateSummary();
          this.updatePagedData();
          this.loading = false;
          this.refreshMonthlyStatus();
        },
        error: () => {
          this.resetTable();
          this.errorMessage = 'No se pudo consultar el forecast mensual.';
          this.loading = false;
          this.refreshMonthlyStatus();
        }
      });
  }

  generateMonthlyForecast(): void {
    this.generating = true;
    this.errorMessage = '';
    this.successMessage = '';

    this.forecastActionsService
      .generateMonthlyForecast(this.selectedChannel, this.startDate, this.endDate)
      .subscribe({
        next: (response: ForecastMonthlyResponse) => {
          this.successMessage = response.message;
          this.generating = false;
          this.refreshMonthlyStatus();
          this.loadMonthlyForecast();
        },
        error: (error) => {
          this.generating = false;
          this.errorMessage = error?.error?.detail || 'No se pudo generar el forecast mensual.';
        }
      });
  }

  onSearch(): void {
    this.refreshMonthlyStatus();
    this.loadMonthlyForecast();
  }

  updatePagedData(): void {
    const startIndex = (this.currentPage - 1) * this.pageSize;
    const endIndex = startIndex + this.pageSize;
    this.pagedForecastIntervals = this.forecastIntervals.slice(startIndex, endIndex);
  }

  previousPage(): void {
    if (this.currentPage > 1) {
      this.currentPage--;
      this.updatePagedData();
    }
  }

  nextPage(): void {
    if (this.currentPage < this.totalPages) {
      this.currentPage++;
      this.updatePagedData();
    }
  }

  goToFirstPage(): void {
    if (this.currentPage !== 1) {
      this.currentPage = 1;
      this.updatePagedData();
    }
  }

  goToLastPage(): void {
    if (this.totalPages > 0 && this.currentPage !== this.totalPages) {
      this.currentPage = this.totalPages;
      this.updatePagedData();
    }
  }

  formatTime(value: string): string {
    if (!value) {
      return '-';
    }

    return value.substring(0, 5);
  }

  getDisplaySlot(item: ForecastIntervalHistoryItem): number {
    return item.slot_index + 1;
  }

  get canGoPrevious(): boolean {
    return this.currentPage > 1;
  }

  get canGoNext(): boolean {
    return this.currentPage < this.totalPages;
  }


  get statusLabel(): string {
    if (this.checkingStatus) {
      return 'Verificando forecast...';
    }

    if (!this.monthlyStatus) {
      return 'Estado no disponible';
    }

    if (this.monthlyStatus.status === 'complete') {
      return 'Forecast mensual completo';
    }

    if (this.monthlyStatus.status === 'partial') {
      return 'Forecast mensual incompleto';
    }

    return 'Forecast mensual no generado';
  }

  get statusClass(): string {
    if (!this.monthlyStatus) {
      return 'status-neutral';
    }

    return `status-${this.monthlyStatus.status}`;
  }

  get canGenerateMonthlyForecast(): boolean {
    return !this.loading && !this.generating && this.monthlyStatus?.status !== 'complete';
  }

  private calculateSummary(): void {
    this.totalForecast = this.forecastIntervals.reduce(
      (sum, item) => sum + Number(item.predicted_value || 0),
      0
    );

    const validAhtValues = this.forecastIntervals
      .map((item) => Number(item.aht || 0))
      .filter((value) => value > 0);

    this.averageAht = validAhtValues.length > 0
      ? validAhtValues.reduce((sum, value) => sum + value, 0) / validAhtValues.length
      : 0;

    this.totalRequiredAgents = this.forecastIntervals.reduce(
      (sum, item) => sum + Number(item.required_agents || 0),
      0
    );
  }

  private resetTable(): void {
    this.forecastIntervals = [];
    this.pagedForecastIntervals = [];
    this.currentPage = 1;
    this.totalPages = 0;
    this.totalRecords = 0;
    this.totalForecast = 0;
    this.averageAht = 0;
    this.totalRequiredAgents = 0;
  }
}
