import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';

import { ExternalVariable } from '../../models/external-variable.model';
import { ExternalVariablesService } from '../../services/external-variables.service';

@Component({
  selector: 'app-external-variables',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './external-variables.component.html',
  styleUrl: './external-variables.component.css'
})
export class ExternalVariablesComponent implements OnInit {
  variables: ExternalVariable[] = [];
  isLoading = false;
  errorMessage = '';
  readonly pageSize = 10;
  currentPage = 1;

  constructor(private readonly externalVariablesService: ExternalVariablesService) {}

  ngOnInit(): void {
    this.loadVariables();
  }

  loadVariables(): void {
    this.isLoading = true;
    this.errorMessage = '';

    this.externalVariablesService.getExternalVariables().subscribe({
      next: (response) => {
        this.variables = response ?? [];
        this.currentPage = 1;
        this.isLoading = false;
      },
      error: (error) => {
        this.errorMessage = error?.error?.detail || 'No se pudieron obtener las variables externas.';
        this.isLoading = false;
      }
    });
  }

  get pagedVariables(): ExternalVariable[] {
    return this.paginate(this.variables, this.currentPage);
  }

  get totalPages(): number {
    return this.getTotalPages(this.variables.length);
  }

  get startItem(): number {
    return this.variables.length === 0 ? 0 : (this.currentPage - 1) * this.pageSize + 1;
  }

  get endItem(): number {
    return Math.min(this.currentPage * this.pageSize, this.variables.length);
  }

  goToPage(page: number): void {
    this.currentPage = Math.min(Math.max(page, 1), this.totalPages);
  }

  private paginate<T>(items: T[], page: number): T[] {
    const start = (page - 1) * this.pageSize;
    return items.slice(start, start + this.pageSize);
  }

  private getTotalPages(totalItems: number): number {
    return Math.max(1, Math.ceil(totalItems / this.pageSize));
  }

  getBadgeClass(type: string): string {
    const normalized = type.toLowerCase();
    if (normalized === 'is_holiday') return 'holiday';
    if (normalized === 'campaign_day') return 'campaign';
    if (normalized === 'absenteeism_rate') return 'absenteeism';
    return 'default';
  }

  getLabel(type: string): string {
    const labels: Record<string, string> = {
      is_holiday: 'Feriado',
      campaign_day: 'Campaña',
      absenteeism_rate: 'Ausentismo'
    };
    return labels[type] ?? type;
  }
}
