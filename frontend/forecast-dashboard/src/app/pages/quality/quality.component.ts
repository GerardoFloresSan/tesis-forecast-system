import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

import { QualityReport, QualityRiskStatus } from '../../models/quality.model';
import { QualityService } from '../../services/quality.service';

interface KeyValueNumber {
  key: string;
  value: number;
}

@Component({
  selector: 'app-quality',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './quality.component.html',
  styleUrl: './quality.component.css'
})
export class QualityComponent {
  report: QualityReport | null = null;
  isLoading = false;
  errorMessage = '';

  readonly pageSize = 10;
  recordsByChannelPage = 1;
  nullsByColumnPage = 1;
  issuesPage = 1;

  constructor(private readonly qualityService: QualityService) {}

  loadReport(): void {
    this.isLoading = true;
    this.errorMessage = '';

    this.qualityService.getQualityReport().subscribe({
      next: (response) => {
        this.report = response;
        this.recordsByChannelPage = 1;
        this.nullsByColumnPage = 1;
        this.issuesPage = 1;
        this.isLoading = false;
      },
      error: (error) => {
        this.errorMessage = error?.error?.detail || 'No se pudo ejecutar el análisis de calidad.';
        this.isLoading = false;
      }
    });
  }

  getRiskStatus(): QualityRiskStatus {
    if (!this.report) return 'Óptimo';
    const risk = Math.max(this.report.missing_percentage, this.report.duplicate_percentage);
    if (risk < 16) return 'Óptimo';
    if (risk <= 20) return 'Aceptable';
    return 'Deficiente';
  }

  getRiskClass(): string {
    const status = this.getRiskStatus();
    if (status === 'Óptimo') return 'optimal';
    if (status === 'Aceptable') return 'acceptable';
    return 'deficient';
  }

  get recordsByChannel(): KeyValueNumber[] {
    return this.objectEntries(this.report?.records_by_channel);
  }

  get nullsByColumn(): KeyValueNumber[] {
    return this.objectEntries(this.report?.nulls_by_column);
  }

  get pagedRecordsByChannel(): KeyValueNumber[] {
    return this.paginate(this.recordsByChannel, this.recordsByChannelPage);
  }

  get pagedNullsByColumn(): KeyValueNumber[] {
    return this.paginate(this.nullsByColumn, this.nullsByColumnPage);
  }

  get pagedIssues(): string[] {
    return this.paginate(this.report?.summary.issues ?? [], this.issuesPage);
  }

  get recordsByChannelTotalPages(): number {
    return this.getTotalPages(this.recordsByChannel.length);
  }

  get nullsByColumnTotalPages(): number {
    return this.getTotalPages(this.nullsByColumn.length);
  }

  get issuesTotalPages(): number {
    return this.getTotalPages(this.report?.summary.issues.length ?? 0);
  }

  goToRecordsByChannelPage(page: number): void {
    this.recordsByChannelPage = this.clampPage(page, this.recordsByChannelTotalPages);
  }

  goToNullsByColumnPage(page: number): void {
    this.nullsByColumnPage = this.clampPage(page, this.nullsByColumnTotalPages);
  }

  goToIssuesPage(page: number): void {
    this.issuesPage = this.clampPage(page, this.issuesTotalPages);
  }

  objectEntries(record: Record<string, number> | null | undefined): KeyValueNumber[] {
    return Object.entries(record ?? {}).map(([key, value]) => ({ key, value }));
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
}
