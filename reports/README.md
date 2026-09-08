# SatQuery AI — Report Generation Engine (`reports/`)

**Module**: `reports/`  
**Shared Contract**: `core.schemas.QueryResponse`, `core.schemas.Artifact`

---

## 1. Overview
The Report Generation Engine enables users, analysts, and judges to export comprehensive, self-contained intelligence analysis reports in three standard formats:
1. **Interactive Standalone HTML Report**: Includes CSS theme, metadata badges, embedded visual evidence (via base64 and artifact links), stage-by-stage execution trace timeline, and print/PDF-ready styling (`window.print()`).
2. **GitHub-Flavored Markdown Report**: Formatted Markdown document ideal for technical documentation, Git commits, PR summaries, and markdown viewers.
3. **Machine-Readable JSON Audit Dump**: Complete serializable JSON object containing query, answer, calibrated confidence, structured evidence, artifact references, operational trace latencies, and limitation disclaimers.

---

## 2. Report Sections & Structure

Every generated report contains:
- **Header**: SatQuery AI title, unique Report ID, correlated Request ID, UTC timestamp.
- **Natural Language Query & Grounded Answer**: Original prompt and synthesized specialist response.
- **Calibrated Confidence**: Formatted percentage score, decimal value, qualitative classification tier (`HIGH`, `MODERATE`, `LOW`, `UNAVAILABLE`), and operational rationale.
- **Ingested Imagery & Task Metadata**: Image count, observed sensor modalities (`optical`, `multispectral`, `sar`), format (`GeoTIFF`, `PNG`), and georeferencing metadata if verified.
- **Visual Grounding Evidence Catalog**: Bounding box coordinates, difference masks, false-color composites, and crop artifacts.
- **Auditable Operational Execution Trace**: Total pipeline latency and duration for every stage (`INPUT_VALIDATED`, `TASK_RESOLVED`, `TOOL_SELECTED`, `INFERENCE_EXECUTED`, etc.).
- **Authentic Operational Limitations**: Non-fabricated disclosure of uncalibrated confidence, low certainty warnings, missing bounding boxes, or sensor resolution boundaries.

---

## 3. Python API Usage

```python
from reports.generator import ReportGenerator
from core.schemas import QueryResponse

# Generate reports from any QueryResponse
json_report = ReportGenerator.generate_json_report(query_response, save_to_disk=True)
md_report = ReportGenerator.generate_markdown_report(query_response, save_to_disk=True)
html_report = ReportGenerator.generate_html_report(query_response, save_to_disk=True)

print(f"Generated HTML report at: {html_report.file_path}")
print(f"Artifact ID: {html_report.artifact.artifact_id}")
```

---

## 4. REST API Endpoints

- `POST /api/v1/reports/generate`: Generates report from a `QueryResponse` payload (`format: "html" | "markdown" | "json"`).
- `GET /api/v1/reports/{report_id}`: Downloads the generated report file directly.
