export type ImageModality = 'optical' | 'multispectral' | 'sar' | 'unknown';
export type ImageFormat = 'geotiff' | 'tiff' | 'png' | 'jpeg';
export type TaskType =
  | 'single_image_vqa'
  | 'single_image_caption'
  | 'single_image_grounding'
  | 'change_analysis'
  | 'change_vqa'
  | 'optical_sar_analysis';

export type EvidenceType =
  | 'bounding_box'
  | 'mask'
  | 'change_map'
  | 'highlighted_image'
  | 'crop'
  | 'heatmap'
  | 'text_evidence';

export type ToolStatus = 'success' | 'partial_success' | 'failed' | 'unsupported';

export type ExecutionStage =
  | 'REQUEST_RECEIVED'
  | 'INPUT_VALIDATED'
  | 'TASK_RESOLVED'
  | 'TOOL_SELECTED'
  | 'MODEL_INITIALIZED'
  | 'INFERENCE_EXECUTED'
  | 'EVIDENCE_GENERATED'
  | 'RESULT_AGGREGATED'
  | 'RESULT_RETURNED'
  | 'ERROR_ENCOUNTERED';

export interface ImageInput {
  image_id: string;
  path_or_uri: string;
  format: ImageFormat;
  modality: ImageModality;
  width?: number;
  height?: number;
  channel_count?: number;
  dtype?: string;
  geospatial?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface TaskIntent {
  task: TaskType;
  confidence: number;
  intent_explanation: string;
  target_features?: string[];
  extracted_parameters?: Record<string, any>;
}

export interface Evidence {
  id: string;
  type: EvidenceType;
  label: string;
  confidence?: number | null;
  data: {
    bbox?: number[];
    format?: string;
    coordinate_system?: string;
    changed_pixel_ratio?: number;
    changed_pixel_count?: number;
    total_pixel_count?: number;
    num_regions?: number;
    t0_image_id?: string;
    t1_image_id?: string;
    [key: string]: any;
  };
  image_id?: string | null;
  metadata?: Record<string, any>;
}

export interface Artifact {
  artifact_id: string;
  name: string;
  type: string;
  uri_or_path: string;
  description: string;
  mime_type?: string | null;
  metadata?: Record<string, any>;
}

export interface ExecutionTraceEntry {
  stage: ExecutionStage;
  component: string;
  status: string;
  timestamp: string;
  duration_ms?: number | null;
  details?: Record<string, any>;
}

export interface TaskPlanStep {
  step_id: string;
  step_index: number;
  task: TaskType;
  tool_name: string;
  purpose: string;
  status: string;
  input_references?: string[];
  dependencies?: string[];
  pass_context_from_previous?: boolean;
}

export interface TaskPlan {
  plan_id: string;
  goal: string;
  steps: TaskPlanStep[];
  is_multi_step: boolean;
  metadata?: Record<string, any>;
}

export interface AgentDecision {
  task: TaskType;
  task_display_name: string;
  image_count: number;
  detected_modalities: ImageModality[];
  selected_specialist: string;
  workflow_summary: string;
  confidence?: number | null;
  why_this_tool: string;
}

export interface SatQueryErrorDetail {
  error_code: string;
  message: string;
  field?: string | null;
  request_id?: string | null;
  details?: Record<string, any>;
}

export interface QueryRequest {
  query: string;
  images: {
    path_or_uri: string;
    format: ImageFormat;
    modality?: ImageModality;
    image_id?: string;
  }[];
  task_hint?: TaskType;
  config?: Record<string, any>;
}

export interface QueryResponse {
  request_id: string;
  query: string;
  resolved_task: TaskType;
  status: ToolStatus;
  answer: string;
  confidence?: number | null;
  evidence: Evidence[];
  artifacts: Artifact[];
  execution_trace: ExecutionTraceEntry[];
  task_intent?: TaskIntent;
  task_plan?: TaskPlan;
  agent_decision?: AgentDecision;
  selected_tools: string[];
  errors: SatQueryErrorDetail[];
  metadata?: Record<string, any>;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded';
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
  registered_tools_count: number;
  tools_health: Record<string, boolean>;
}

export interface SupportedTask {
  task: TaskType;
  description: string;
  required_images: number;
  supported_modalities: string[];
}

export interface ToolMetadata {
  name: string;
  description: string;
  version: string;
  supported_tasks: TaskType[];
  required_modalities: ImageModality[];
  min_images: number;
  max_images: number;
  author_or_division: string;
  metadata: Record<string, any>;
}

export interface ReportGenerationRequest {
  response: QueryResponse;
  format: 'html' | 'markdown' | 'json';
}

export interface ReportGenerationResponse {
  status: string;
  report_id: string;
  format: string;
  download_url: string;
  file_name: string;
  artifact: Artifact;
  content_preview: string;
}

export interface BenchmarkDefinition {
  id: string;
  name: string;
  description: string;
  metrics: string[];
}

export interface BenchmarkRunRequest {
  benchmark: string;
  predictions: Record<string, any>[];
  ground_truths: Record<string, any>[];
  config?: Record<string, any>;
}

export interface BenchmarkMetricResult {
  name?: string;
  metric_type?: string;
  raw_score?: number;
  normalized_score?: number;
  sample_count?: number;
  interpretation?: string;
  metadata?: Record<string, any>;
  score?: number;
  num_samples?: number;
}

export interface BenchmarkRunResponse {
  status: string;
  benchmark: string;
  samples_evaluated: number;
  aggregate_normalized_score: number;
  aggregate_raw_score: number;
  metrics: Record<string, BenchmarkMetricResult>;
  per_category_scores: Record<string, number>;
  scoreboard_markdown: string;
}

export interface PresetScenario {
  id: string;
  name: string;
  category: 'Single Image' | 'Bi-Temporal' | 'Cross-Modal' | 'Multi-Tool';
  subtitle: string;
  query: string;
  source: string;
  sensor: string;
  license: string;
  images: {
    path_or_uri: string;
    format: ImageFormat;
    modality: ImageModality;
    role: string;
    name: string;
  }[];
  expectedTask: TaskType;
  iconName: string;
}

