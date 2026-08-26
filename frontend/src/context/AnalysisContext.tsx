import React, { createContext, useContext, useState, useEffect } from 'react';
import {
  Artifact,
  Evidence,
  PresetScenario,
  QueryResponse,
  SystemHealth,
  TaskType,
} from '../types/api';
import { api } from '../services/api';

export const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'demo-a',
    name: 'Demo A: Single-Image VQA',
    category: 'Single Image',
    subtitle: 'Copernicus Sentinel-2 MSI Optical VQA',
    query: 'What is the dominant land cover and infrastructure in this scene?',
    source: 'Copernicus Sentinel-2 (ESA)',
    sensor: 'MSI 10m True-Color RGB',
    license: 'CC BY-SA 3.0 IGO',
    expectedTask: 'single_image_vqa',
    iconName: 'Eye',
    images: [
      {
        path_or_uri: 'demo_assets/demo_optical_single.png',
        format: 'png',
        modality: 'optical',
        role: 'Sentinel-2 MSI Optical',
        name: 'demo_optical_single.png',
      },
    ],
  },
  {
    id: 'demo-b',
    name: 'Demo B: Spatial Grounding',
    category: 'Single Image',
    subtitle: 'USGS High-Res Aerial Airfield Localization',
    query: 'Where is the airport runway and apron in this image?',
    source: 'USGS Aerial Orthoimagery',
    sensor: '0.5m High-Resolution NAIP',
    license: 'Public Domain (USGS)',
    expectedTask: 'single_image_grounding',
    iconName: 'Crosshair',
    images: [
      {
        path_or_uri: 'demo_assets/demo_airport_grounding.png',
        format: 'png',
        modality: 'optical',
        role: 'USGS Airport Orthoimagery',
        name: 'demo_airport_grounding.png',
      },
    ],
  },
  {
    id: 'demo-c',
    name: 'Demo C: Bi-Temporal Change',
    category: 'Bi-Temporal',
    subtitle: 'LEVIR-CD High-Res Surface Change Intelligence',
    query: 'What changed between these two acquisition dates?',
    source: 'LEVIR-CD Benchmark (Beihang Univ)',
    sensor: '0.5m Google Earth / WorldView',
    license: 'Academic Research Open Access',
    expectedTask: 'change_vqa',
    iconName: 'GitCompare',
    images: [
      {
        path_or_uri: 'demo_assets/demo_change_t0.png',
        format: 'png',
        modality: 'optical',
        role: 'T0 — Baseline Acquisition',
        name: 'demo_change_t0.png',
      },
      {
        path_or_uri: 'demo_assets/demo_change_t1.png',
        format: 'png',
        modality: 'optical',
        role: 'T1 — Developed Acquisition',
        name: 'demo_change_t1.png',
      },
    ],
  },
  {
    id: 'demo-d',
    name: 'Demo D: Optical-SAR Fusion',
    category: 'Cross-Modal',
    subtitle: 'Sentinel-2 Optical + Sentinel-1 C-Band SAR',
    query: 'Use optical and SAR images together to identify structures beneath clouds.',
    source: 'Copernicus Sentinel-1/2 (ESA)',
    sensor: 'MSI Optical + C-SAR GRD (Mock Model)',
    license: 'CC BY-SA 3.0 IGO',
    expectedTask: 'optical_sar_analysis',
    iconName: 'Layers',
    images: [
      {
        path_or_uri: 'demo_assets/demo_optical_cross.png',
        format: 'png',
        modality: 'optical',
        role: 'Sentinel-2 Clouded Optical',
        name: 'demo_optical_cross.png',
      },
      {
        path_or_uri: 'demo_assets/demo_sar_cross.tif',
        format: 'tiff',
        modality: 'sar',
        role: 'Sentinel-1 SAR Radar Amplitude',
        name: 'demo_sar_cross.tif',
      },
    ],
  },
  {
    id: 'demo-e',
    name: 'Demo E: Multi-Tool Workflow',
    category: 'Multi-Tool',
    subtitle: 'LEVIR-CD Change Detection → Scene Analysis',
    query: 'What changed, where did it happen, and what is present in the change?',
    source: 'LEVIR-CD Dataset',
    sensor: '0.5m Bi-Temporal Satellite',
    license: 'Academic Research Open Access',
    expectedTask: 'change_vqa',
    iconName: 'Workflow',
    images: [
      {
        path_or_uri: 'demo_assets/demo_change_t0.png',
        format: 'png',
        modality: 'optical',
        role: 'T0 — Pre-Development',
        name: 'demo_change_t0.png',
      },
      {
        path_or_uri: 'demo_assets/demo_change_t1.png',
        format: 'png',
        modality: 'optical',
        role: 'T1 — Post-Development',
        name: 'demo_change_t1.png',
      },
    ],
  },
];

export interface UploadedFileItem {
  id: string;
  file: File;
  previewUrl: string;
  role: string;
  modality: 'optical' | 'sar' | 'multispectral';
}

interface AnalysisContextType {
  // Input state
  inputMode: 'preset' | 'upload';
  setInputMode: (mode: 'preset' | 'upload') => void;
  selectedPreset: PresetScenario | null;
  selectPreset: (preset: PresetScenario) => void;
  uploadedFiles: UploadedFileItem[];
  addUploadedFiles: (files: File[]) => void;
  removeUploadedFile: (id: string) => void;
  updateFileRole: (id: string, role: string) => void;
  updateFileModality: (id: string, modality: 'optical' | 'sar' | 'multispectral') => void;
  clearUploads: () => void;

  // Query state
  query: string;
  setQuery: (q: string) => void;
  isCustomQuery: boolean;

  // Execution state
  isLoading: boolean;
  loadingStage: string;
  pipelineStages: { stage: string; status: 'pending' | 'active' | 'completed' | 'failed' }[];
  result: QueryResponse | null;
  error: string | null;
  executeAnalysis: (customQueryOverride?: string) => Promise<void>;

  // Selection/Inspection state
  activeArtifact: Artifact | null;
  setActiveArtifact: (art: Artifact | null) => void;
  activeEvidence: Evidence | null;
  setActiveEvidence: (ev: Evidence | null) => void;

  // System status
  systemHealth: SystemHealth | null;
  refreshHealth: () => Promise<void>;
}

const AnalysisContext = createContext<AnalysisContextType | null>(null);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [inputMode, setInputMode] = useState<'preset' | 'upload'>('preset');
  const [selectedPreset, setSelectedPreset] = useState<PresetScenario | null>(PRESET_SCENARIOS[0]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileItem[]>([]);
  const [query, setQuery] = useState<string>(PRESET_SCENARIOS[0].query);
  const [isCustomQuery, setIsCustomQuery] = useState<boolean>(false);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStage, setLoadingStage] = useState<string>('Idle');
  const [pipelineStages, setPipelineStages] = useState<{ stage: string; status: 'pending' | 'active' | 'completed' | 'failed' }[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<Evidence | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);

  // Initial health check
  useEffect(() => {
    refreshHealth();
  }, []);

  const refreshHealth = async () => {
    try {
      const health = await api.getHealth();
      setSystemHealth(health);
    } catch (e) {
      setSystemHealth({
        status: 'degraded',
        app_name: 'SatQuery AI',
        version: '0.1.0',
        environment: 'development',
        timestamp: new Date().toISOString(),
        registered_tools_count: 0,
        tools_health: {},
      });
    }
  };

  const selectPreset = (preset: PresetScenario) => {
    setSelectedPreset(preset);
    setInputMode('preset');
    setQuery(preset.query);
    setIsCustomQuery(false);
  };

  const addUploadedFiles = (files: File[]) => {
    const newItems: UploadedFileItem[] = files.map((f, idx) => {
      const ext = f.name.split('.').pop()?.toLowerCase();
      const isSar = ext === 'tif' || ext === 'tiff' || f.name.toLowerCase().includes('sar');
      const currentCount = uploadedFiles.length + idx;
      
      let defaultRole = 'Primary Image';
      if (currentCount === 0) defaultRole = 'T0 — Baseline / Optical';
      else if (currentCount === 1) defaultRole = isSar ? 'SAR Radar Layer' : 'T1 — Follow-Up';

      return {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        file: f,
        previewUrl: URL.createObjectURL(f),
        role: defaultRole,
        modality: isSar ? 'sar' : 'optical',
      };
    });

    setUploadedFiles((prev) => [...prev, ...newItems]);
    setInputMode('upload');
  };

  const removeUploadedFile = (id: string) => {
    setUploadedFiles((prev) => {
      const item = prev.find((i) => i.id === id);
      if (item) URL.revokeObjectURL(item.previewUrl);
      return prev.filter((i) => i.id !== id);
    });
  };

  const updateFileRole = (id: string, role: string) => {
    setUploadedFiles((prev) => prev.map((f) => (f.id === id ? { ...f, role } : f)));
  };

  const updateFileModality = (id: string, modality: 'optical' | 'sar' | 'multispectral') => {
    setUploadedFiles((prev) => prev.map((f) => (f.id === id ? { ...f, modality } : f)));
  };

  const clearUploads = () => {
    uploadedFiles.forEach((f) => URL.revokeObjectURL(f.previewUrl));
    setUploadedFiles([]);
  };

  const executeAnalysis = async (customQueryOverride?: string) => {
    const activeQuery = customQueryOverride !== undefined ? customQueryOverride : query;
    if (!activeQuery.trim()) {
      setError('Please enter a query or select a preset scenario.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setLoadingStage('Initializing agent orchestration...');
    setPipelineStages([
      { stage: 'Validating image geometry & CRS', status: 'active' },
      { stage: 'Resolving intent & specialist routing', status: 'pending' },
      { stage: 'Executing domain vision-language neural model', status: 'pending' },
      { stage: 'Generating grounding evidence & spatial maps', status: 'pending' },
      { stage: 'Synthesizing evidence-grounded response', status: 'pending' },
    ]);

    try {
      let res: QueryResponse;

      if (inputMode === 'upload' && uploadedFiles.length > 0) {
        // Multipart Upload Pipeline
        const formData = new FormData();
        formData.append('query', activeQuery);
        
        const modalities = uploadedFiles.map((f) => f.modality);
        formData.append('modalities', JSON.stringify(modalities));

        uploadedFiles.forEach((f) => {
          formData.append('files', f.file);
        });

        setLoadingStage('Uploading rasters & executing agent pipeline...');
        setPipelineStages((prev) =>
          prev.map((s, i) => (i === 0 ? { ...s, status: 'completed' } : i === 1 ? { ...s, status: 'active' } : s))
        );

        res = await api.submitQueryMultipart(formData);
      } else {
        // Preset Pipeline
        if (!selectedPreset) {
          throw new Error('No preset scenario selected.');
        }

        const payloadImages = selectedPreset.images.map((img) => ({
          path_or_uri: img.path_or_uri,
          format: img.format,
          modality: img.modality,
        }));

        setLoadingStage(`Executing agent plan for '${selectedPreset.name}'...`);
        setPipelineStages((prev) =>
          prev.map((s, i) => (i === 0 ? { ...s, status: 'completed' } : i === 1 ? { ...s, status: 'active' } : s))
        );

        res = await api.submitQuery({
          query: activeQuery,
          images: payloadImages,
          task_hint: selectedPreset.expectedTask,
        });
      }

      setPipelineStages((prev) => prev.map((s) => ({ ...s, status: 'completed' })));
      setResult(res);

      // Select high-priority visual artifact automatically
      if (res.artifacts && res.artifacts.length > 0) {
        const visualPriority = (name: string) => {
          const n = (name || '').toLowerCase();
          if (n.includes('bitemporal_change_composite') || n.includes('optical_sar_fusion')) return 10;
          if (n.includes('annotated_grounding')) return 8;
          if (n.includes('heatmap')) return 6;
          if (n.includes('crop')) return 4;
          if (n.includes('mask')) return 2;
          return 1;
        };

        const sortedArts = [...res.artifacts].sort((a, b) => visualPriority(b.name) - visualPriority(a.name));
        setActiveArtifact(sortedArts[0]);
      } else {
        setActiveArtifact(null);
      }

      if (res.evidence && res.evidence.length > 0) {
        setActiveEvidence(res.evidence[0]);
      } else {
        setActiveEvidence(null);
      }
    } catch (err: any) {
      const errMsg = err.response?.data?.detail || err.message || 'Error occurred while running analysis.';
      setError(errMsg);
      setPipelineStages((prev) => prev.map((s) => (s.status === 'active' ? { ...s, status: 'failed' } : s)));
    } finally {
      setIsLoading(false);
      setLoadingStage('Completed');
    }
  };

  return (
    <AnalysisContext.Provider
      value={{
        inputMode,
        setInputMode,
        selectedPreset,
        selectPreset,
        uploadedFiles,
        addUploadedFiles,
        removeUploadedFile,
        updateFileRole,
        updateFileModality,
        clearUploads,
        query,
        setQuery: (q) => {
          setQuery(q);
          setIsCustomQuery(true);
        },
        isCustomQuery,
        isLoading,
        loadingStage,
        pipelineStages,
        result,
        error,
        executeAnalysis,
        activeArtifact,
        setActiveArtifact,
        activeEvidence,
        setActiveEvidence,
        systemHealth,
        refreshHealth,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};
