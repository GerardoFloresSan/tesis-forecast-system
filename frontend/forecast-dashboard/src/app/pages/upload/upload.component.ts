import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

import { UploadResponse } from '../../models/upload.model';
import { UploadService } from '../../services/upload.service';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css'
})
export class UploadComponent {
  selectedFile: File | null = null;
  isDragging = false;
  isLoading = false;
  result: UploadResponse | null = null;
  errorMessage = '';

  private readonly allowedExtensions = ['xlsx', 'xls', 'csv'];

  constructor(private readonly uploadService: UploadService) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.setFile(file);
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
    const file = event.dataTransfer?.files?.[0] ?? null;
    this.setFile(file);
  }

  upload(): void {
    if (!this.selectedFile || this.isLoading) return;

    this.isLoading = true;
    this.result = null;
    this.errorMessage = '';

    this.uploadService.uploadFile(this.selectedFile).subscribe({
      next: (response) => {
        this.result = response;
        this.isLoading = false;
      },
      error: (error) => {
        this.errorMessage = error?.error?.detail || error?.error?.message || 'No se pudo cargar el archivo. Verifica el formato y vuelve a intentar.';
        this.isLoading = false;
      }
    });
  }

  clear(): void {
    this.selectedFile = null;
    this.result = null;
    this.errorMessage = '';
  }

  private setFile(file: File | null): void {
    this.result = null;
    this.errorMessage = '';

    if (!file) {
      this.selectedFile = null;
      return;
    }

    const extension = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (!this.allowedExtensions.includes(extension)) {
      this.selectedFile = null;
      this.errorMessage = 'Formato no permitido. Usa archivos Excel (.xlsx, .xls) o CSV.';
      return;
    }

    this.selectedFile = file;
  }
}
