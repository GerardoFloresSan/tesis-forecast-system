from __future__ import annotations

from datetime import datetime
from typing import Any

from fpdf import FPDF


class QualityReportPdf(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Forecast System - Reporte de calidad de datos", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, f"Pagina {self.page_no()}", align="C")


class PdfBuilder:
    def __init__(self) -> None:
        self.pdf = QualityReportPdf()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_margins(left=12, top=14, right=12)
        self.pdf.add_page()

    def section_title(self, title: str) -> None:
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "B", 11)
        self.pdf.set_fill_color(236, 242, 248)
        self.pdf.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(1)

    def paragraph(self, text: str) -> None:
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.set_font("Helvetica", size=9)
        self.pdf.multi_cell(0, 5, text)

    def kv_line(self, label: str, value: Any) -> None:
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.cell(64, 6, f"{label}:")
        self.pdf.set_font("Helvetica", size=9)
        self.pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def simple_table(self, rows: list[tuple[str, Any]], first_col_width: float = 85) -> None:
        second_col_width = self.pdf.epw - first_col_width
        self.pdf.set_font("Helvetica", "B", 9)
        for label, value in rows:
            self.pdf.cell(first_col_width, 7, str(label), border=1)
            self.pdf.set_font("Helvetica", size=9)
            self.pdf.cell(second_col_width, 7, str(value), border=1, new_x="LMARGIN", new_y="NEXT")
            self.pdf.set_font("Helvetica", "B", 9)

    def bullet_list(self, items: list[str]) -> None:
        self.pdf.set_font("Helvetica", size=9)
        for item in items:
            self.pdf.set_x(self.pdf.l_margin)
            self.pdf.multi_cell(0, 5, f"- {item}")



def _fmt_percentage(value: Any) -> str:
    return f"{float(value):.2f}%"


def _join_dates(values: list[str], max_items: int = 10) -> str:
    if not values:
        return "Ninguno"
    if len(values) <= max_items:
        return ", ".join(values)
    return ", ".join(values[:max_items]) + f" ... (+{len(values) - max_items} mas)"



def generate_quality_pdf(report_data: dict[str, Any]) -> bytes:
    builder = PdfBuilder()
    pdf = builder.pdf

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = report_data.get("summary", {})
    date_range = report_data.get("date_range", {})
    duplicate_keys = report_data.get("duplicate_keys", {})
    interval_summary = report_data.get("intervals", {})
    days_without_data = report_data.get("days_without_data", {})

    builder.paragraph(
        "Documento generado automaticamente a partir del endpoint de control de calidad. "
        "Resume completitud, duplicados, intervalos observados y hallazgos relevantes "
        "para la tabla historical_interactions."
    )
    builder.kv_line("Fecha de generacion", generated_at)
    builder.kv_line("Estado general", summary.get("status", "N/D"))

    builder.section_title("Metricas principales")
    builder.simple_table(
        [
            ("Total de registros", report_data.get("total_records", 0)),
            ("Porcentaje de nulos", _fmt_percentage(report_data.get("missing_percentage", 0))),
            ("Porcentaje de duplicados", _fmt_percentage(report_data.get("duplicate_percentage", 0))),
            ("Porcentaje valido", _fmt_percentage(report_data.get("valid_percentage", 0))),
            (
                "Rango de fechas",
                f"{date_range.get('start_date') or 'N/D'} a {date_range.get('end_date') or 'N/D'}",
            ),
            ("Dias cubiertos", date_range.get("total_days", 0)),
            ("Canales detectados", ", ".join(report_data.get("detected_channels", [])) or "Ninguno"),
        ]
    )

    builder.section_title("Nulos por columna")
    null_rows = [(column, value) for column, value in report_data.get("nulls_by_column", {}).items()]
    if null_rows:
        builder.simple_table(null_rows)
    else:
        builder.paragraph("No se encontraron columnas para evaluar.")

    builder.section_title("Duplicados por clave logica")
    builder.simple_table(
        [
            ("Grupos duplicados", duplicate_keys.get("duplicate_groups", 0)),
            ("Registros duplicados", duplicate_keys.get("duplicate_records", 0)),
        ]
    )
    duplicate_sample = duplicate_keys.get("sample", [])
    if duplicate_sample:
        builder.paragraph("Muestra de claves duplicadas detectadas:")
        for item in duplicate_sample:
            builder.paragraph(
                f"- Fecha {item.get('interaction_date')}, intervalo {item.get('interval_time')}, "
                f"canal {item.get('channel')}, ocurrencias {item.get('occurrences')}"
            )
    else:
        builder.paragraph("No se detectaron claves duplicadas.")

    builder.section_title("Resumen de intervalos")
    builder.simple_table(
        [
            ("Total de intervalos invalidos", interval_summary.get("total_invalid_intervals", 0)),
            ("Total de intervalos faltantes", interval_summary.get("total_missing_intervals", 0)),
        ]
    )
    channel_rows = interval_summary.get("channels", [])
    if channel_rows:
        for channel_info in channel_rows:
            builder.paragraph(
                f"Canal {channel_info.get('channel')}: cadencia {channel_info.get('cadence_minutes') or 'N/D'} min, "
                f"fechas analizadas {channel_info.get('dates_analyzed', 0)}, "
                f"intervalos invalidos {channel_info.get('invalid_intervals_count', 0)}, "
                f"intervalos faltantes {channel_info.get('missing_intervals_count', 0)}, "
                f"dias con incidencias {channel_info.get('days_with_issues', 0)}."
            )
    else:
        builder.paragraph("No hay informacion de intervalos para mostrar.")

    builder.section_title("Dias sin data")
    builder.simple_table(
        [
            ("Cantidad de dias sin data", days_without_data.get("count", 0)),
            ("Fechas detectadas", _join_dates(days_without_data.get("dates", []))),
        ],
        first_col_width=70,
    )
    for channel, values in days_without_data.get("by_channel", {}).items():
        builder.paragraph(f"- {channel}: {_join_dates(values)}")

    builder.section_title("Hallazgos y estado general")
    issues = summary.get("issues", []) or ["No se registraron observaciones."]
    builder.bullet_list([str(item) for item in issues])

    return bytes(pdf.output())
