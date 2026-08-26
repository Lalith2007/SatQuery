"""SatQuery AI Division 5: Multi-Format Report Generation Package."""

from reports.generator import GeneratedReport, ReportGenerator
from reports.templates import HTML_REPORT_TEMPLATE

__all__ = [
    "GeneratedReport",
    "ReportGenerator",
    "HTML_REPORT_TEMPLATE",
]
