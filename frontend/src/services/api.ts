import axios, { AxiosInstance } from 'axios';
import {
  BenchmarkDefinition,
  BenchmarkRunRequest,
  BenchmarkRunResponse,
  QueryRequest,
  QueryResponse,
  ReportGenerationRequest,
  ReportGenerationResponse,
  SupportedTask,
  SystemHealth,
  ToolMetadata,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      timeout: 60000,
      headers: {
        'Accept': 'application/json',
      },
    });
  }

  // Health check
  async getHealth(): Promise<SystemHealth> {
    const res = await this.client.get<SystemHealth>('/health');
    return res.data;
  }

  // Tasks catalog
  async getTasks(): Promise<SupportedTask[]> {
    const res = await this.client.get<{ tasks: SupportedTask[] }>('/api/v1/tasks');
    return res.data.tasks;
  }

  // Registered specialist tools
  async getTools(): Promise<ToolMetadata[]> {
    const res = await this.client.get<ToolMetadata[]>('/api/v1/tools');
    return res.data;
  }

  // Submit structured JSON query (Gradio Server queued API with REST fallback)
  async submitQuery(req: QueryRequest): Promise<QueryResponse> {
    try {
      const imagePaths = req.images.map((img) => img.path_or_uri);
      const taskHint = req.task_hint || 'single_image_vqa';
      
      const callRes = await this.client.post<{ event_id: string }>('/gradio_api/call/predict', {
        data: [req.query, taskHint, JSON.stringify(imagePaths)],
      });
      const eventId = callRes.data?.event_id;
      if (eventId) {
        const streamRes = await fetch(`${API_BASE}/gradio_api/call/predict/${eventId}`);
        if (streamRes.ok) {
          const reader = streamRes.body?.getReader();
          const decoder = new TextDecoder('utf-8');
          if (reader) {
            let buffer = '';
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';
              for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data:')) {
                  const jsonStr = trimmed.replace(/^data:\s*/, '');
                  try {
                    const parsed = JSON.parse(jsonStr);
                    if (Array.isArray(parsed) && parsed.length > 0) {
                      return parsed[0] as QueryResponse;
                    }
                  } catch {
                    // partial or keepalive
                  }
                }
              }
            }
          }
        }
      }
    } catch (err) {
      console.warn('Gradio Server queue submission fell back to REST:', err);
    }

    const res = await this.client.post<QueryResponse>('/api/v1/query', req);
    return res.data;
  }

  // Submit multipart form query with uploaded raster files
  async submitQueryMultipart(formData: FormData): Promise<QueryResponse> {
    const res = await this.client.post<QueryResponse>('/api/v1/query/multipart', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  }

  // Artifact URL constructor
  getArtifactUrl(artifactIdOrPath: string): string {
    if (!artifactIdOrPath) return '';
    if (artifactIdOrPath.startsWith('http://') || artifactIdOrPath.startsWith('https://')) {
      return artifactIdOrPath;
    }
    // Clean identifier (strip path prefixes if just filename/id)
    const cleanId = artifactIdOrPath.split('/').pop() || artifactIdOrPath;
    return `${API_BASE}/api/v1/artifacts/${encodeURIComponent(cleanId)}`;
  }

  // Report generation
  async generateReport(req: ReportGenerationRequest): Promise<ReportGenerationResponse> {
    const res = await this.client.post<ReportGenerationResponse>('/api/v1/reports/generate', req);
    return res.data;
  }

  // Report direct download URL
  getReportDownloadUrl(reportId: string): string {
    return `${API_BASE}/api/v1/reports/${encodeURIComponent(reportId)}`;
  }

  // Benchmark definitions
  async getBenchmarks(): Promise<BenchmarkDefinition[]> {
    const res = await this.client.get<{ supported_benchmarks: BenchmarkDefinition[] }>('/api/v1/evaluation/benchmarks');
    return res.data.supported_benchmarks;
  }

  // Benchmark evaluation runner
  async runBenchmark(req: BenchmarkRunRequest): Promise<BenchmarkRunResponse> {
    const res = await this.client.post<BenchmarkRunResponse>('/api/v1/evaluation/run', req);
    return res.data;
  }

  // Dev tool swap
  async swapTool(targetTool: string, useAlternate: boolean): Promise<any> {
    const res = await this.client.post('/api/v1/tools/swap', {
      target_tool: targetTool,
      use_alternate: useAlternate,
    });
    return res.data;
  }
}

export const api = new ApiService();
