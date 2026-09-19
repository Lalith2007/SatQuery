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
import { DEMO_MANIFEST, DemoSampleItem, DemoManifestData } from '../data/demoManifest';

export const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'demo-a',
    name: 'Demo A: Single-Image VQA',
    category: 'Single Image',
    subtitle: 'Copernicus Sentinel-2 MSI Optical VQA (15 Real Scenes)',
    query: DEMO_MANIFEST.demos.demo_a.samples[0].query,
    source: 'Copernicus Sentinel-2 (ESA) / RSVQA-LR Val Split',
    sensor: 'MSI 10m True-Color RGB',
    license: 'CC BY-SA 3.0 IGO',
    expectedTask: 'single_image_vqa',
    iconName: 'Eye',
    images: [
      {
        path_or_uri: 'demo_assets/demo_a_vqa/vqa_01.png',
        format: 'png',
        modality: 'optical',
        role: 'Sentinel-2 MSI Optical Scene #1',
        name: 'vqa_01.png',
      },
    ],
  },
  {
    id: 'demo-b',
    name: 'Demo B: Spatial Grounding',
    category: 'Single Image',
    subtitle: 'High-Res Aerial & Satellite Feature Localization (15 Real Scenes)',
    query: DEMO_MANIFEST.demos.demo_b.samples[0].query,
    source: 'LEVIR-CD Test Partition (0.5m GSD)',
    sensor: '0.5m High-Resolution Optical',
    license: 'Academic Research Open Access',
    expectedTask: 'single_image_grounding',
    iconName: 'Crosshair',
    images: [
      {
        path_or_uri: 'demo_assets/demo_b_grounding/grounding_01.png',
        format: 'png',
        modality: 'optical',
        role: 'LEVIR-CD Orthoimagery Scene #1',
        name: 'grounding_01.png',
      },
    ],
  },
  {
    id: 'demo-c',
    name: 'Demo C: Bi-Temporal Change',
    category: 'Bi-Temporal',
    subtitle: 'LEVIR-CD Building Construction Detection (15 Real Pairs)',
    query: DEMO_MANIFEST.demos.demo_c.samples[0].query,
    source: 'LEVIR-CD Benchmark Test Split (Beihang Univ)',
    sensor: '0.5m Google Earth / WorldView',
    license: 'Academic Research Open Access',
    expectedTask: 'change_vqa',
    iconName: 'GitCompare',
    images: [
      {
        path_or_uri: 'demo_assets/demo_c_change/change_01_t0.png',
        format: 'png',
        modality: 'optical',
        role: 'T0 — Baseline Acquisition',
        name: 'change_01_t0.png',
      },
      {
        path_or_uri: 'demo_assets/demo_c_change/change_01_t1.png',
        format: 'png',
        modality: 'optical',
        role: 'T1 — Developed Acquisition',
        name: 'change_01_t1.png',
      },
    ],
  },
  {
    id: 'demo-d',
    name: 'Demo D: Optical-SAR Fusion',
    category: 'Cross-Modal',
    subtitle: 'WHU-OPT-SAR Co-Registered Cross-Modal Intelligence (15 Real Pairs)',
    query: DEMO_MANIFEST.demos.demo_d.samples[0].query,
    source: 'WHU-OPT-SAR Test Split (Wuhan Univ)',
    sensor: '0.55m Optical + SAR Backscatter GeoTIFF',
    license: 'Academic Research Open Access',
    expectedTask: 'optical_sar_analysis',
    iconName: 'Layers',
    images: [
      {
        path_or_uri: 'demo_assets/demo_d_optical_sar/cross_01_opt.png',
        format: 'png',
        modality: 'optical',
        role: 'High-Res Optical RGB',
        name: 'cross_01_opt.png',
      },
      {
        path_or_uri: 'demo_assets/demo_d_optical_sar/cross_01_sar.tif',
        format: 'tiff',
        modality: 'sar',
        role: 'SAR Amplitude GeoTIFF',
        name: 'cross_01_sar.tif',
      },
    ],
  },
  {
    id: 'demo-e',
    name: 'Demo E: Multi-Tool Workflow',
    category: 'Multi-Tool',
    subtitle: 'Autonomous TinyCD Change Detection → Qwen-VL Reasoning (15 Real Triplets)',
    query: DEMO_MANIFEST.demos.demo_e.samples[0].query,
    source: 'LEVIR-CD / CDVQA Test Evidence',
    sensor: '0.5m Bi-Temporal Multi-Resolution Crops',
    license: 'Academic Research Open Access',
    expectedTask: 'change_vqa',
    iconName: 'Workflow',
    images: [
      {
        path_or_uri: 'demo_assets/demo_e_workflow/workflow_01_t0.png',
        format: 'png',
        modality: 'optical',
        role: 'T0 — Focused Region Crop',
        name: 'workflow_01_t0.png',
      },
      {
        path_or_uri: 'demo_assets/demo_e_workflow/workflow_01_t1.png',
        format: 'png',
        modality: 'optical',
        role: 'T1 — Focused Region Crop',
        name: 'workflow_01_t1.png',
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
  selectedSceneIndex: number;
  selectSceneIndex: (index: number) => void;
  demoManifest: DemoManifestData;
  activeSceneSample: DemoSampleItem | null;
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

function buildPresetForScene(basePreset: PresetScenario, sceneIndex: number): PresetScenario {
  const trackKey = basePreset.id.replace('-', '_') as 'demo_a' | 'demo_b' | 'demo_c' | 'demo_d' | 'demo_e';
  const track = DEMO_MANIFEST.demos[trackKey];
  if (!track || !track.samples[sceneIndex]) return basePreset;

  const sample = track.samples[sceneIndex];
  const num = String(sceneIndex + 1).padStart(2, '0');

  let images = basePreset.images;
  if (trackKey === 'demo_a') {
    images = [
      {
        path_or_uri: sample.relative_path || `demo_assets/demo_a_vqa/vqa_${num}.png`,
        format: 'png',
        modality: 'optical',
        role: `Sentinel-2 MSI Scene #${sceneIndex + 1}`,
        name: sample.file_name || `vqa_${num}.png`,
      },
    ];
  } else if (trackKey === 'demo_b') {
    images = [
      {
        path_or_uri: sample.relative_path || `demo_assets/demo_b_grounding/grounding_${num}.png`,
        format: 'png',
        modality: 'optical',
        role: `LEVIR-CD Orthoimagery Scene #${sceneIndex + 1}`,
        name: sample.file_name || `grounding_${num}.png`,
      },
    ];
  } else if (trackKey === 'demo_c') {
    images = [
      {
        path_or_uri: sample.t0_path || `demo_assets/demo_c_change/change_${num}_t0.png`,
        format: 'png',
        modality: 'optical',
        role: `T0 Acquisition (${sample.scene_id})`,
        name: sample.t0_file || `change_${num}_t0.png`,
      },
      {
        path_or_uri: sample.t1_path || `demo_assets/demo_c_change/change_${num}_t1.png`,
        format: 'png',
        modality: 'optical',
        role: `T1 Acquisition (${sample.scene_id})`,
        name: sample.t1_file || `change_${num}_t1.png`,
      },
    ];
  } else if (trackKey === 'demo_d') {
    images = [
      {
        path_or_uri: sample.optical_path || `demo_assets/demo_d_optical_sar/cross_${num}_opt.png`,
        format: 'png',
        modality: 'optical',
        role: `Optical RGB (${sample.tile_stem})`,
        name: sample.optical_file || `cross_${num}_opt.png`,
      },
      {
        path_or_uri: sample.sar_path || `demo_assets/demo_d_optical_sar/cross_${num}_sar.tif`,
        format: 'tiff',
        modality: 'sar',
        role: `SAR GeoTIFF (${sample.tile_stem})`,
        name: sample.sar_file || `cross_${num}_sar.tif`,
      },
    ];
  } else if (trackKey === 'demo_e') {
    images = [
      {
        path_or_uri: sample.t0_path || `demo_assets/demo_e_workflow/workflow_${num}_t0.png`,
        format: 'png',
        modality: 'optical',
        role: `T0 Region Crop (${sample.scene_id})`,
        name: sample.t0_file || `workflow_${num}_t0.png`,
      },
      {
        path_or_uri: sample.t1_path || `demo_assets/demo_e_workflow/workflow_${num}_t1.png`,
        format: 'png',
        modality: 'optical',
        role: `T1 Region Crop (${sample.scene_id})`,
        name: sample.t1_file || `workflow_${num}_t1.png`,
      },
    ];
  }

  return {
    ...basePreset,
    query: sample.query,
    images,
  };
}

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [inputMode, setInputMode] = useState<'preset' | 'upload'>('preset');
  const [selectedPreset, setSelectedPreset] = useState<PresetScenario | null>(PRESET_SCENARIOS[0]);
  const [selectedSceneIndex, setSelectedSceneIndex] = useState<number>(0);
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
    const updatedPreset = buildPresetForScene(preset, 0);
    setSelectedPreset(updatedPreset);
    setSelectedSceneIndex(0);
    setInputMode('preset');
    setQuery(updatedPreset.query);
    setIsCustomQuery(false);
  };

  const selectSceneIndex = (index: number) => {
    if (!selectedPreset) return;
    const updatedPreset = buildPresetForScene(selectedPreset, index);
    setSelectedPreset(updatedPreset);
    setSelectedSceneIndex(index);
    setInputMode('preset');
    setQuery(updatedPreset.query);
    setIsCustomQuery(false);
  };

  const getActiveSceneSample = (): DemoSampleItem | null => {
    if (!selectedPreset) return null;
    const trackKey = selectedPreset.id.replace('-', '_') as 'demo_a' | 'demo_b' | 'demo_c' | 'demo_d' | 'demo_e';
    const track = DEMO_MANIFEST.demos[trackKey];
    return track?.samples[selectedSceneIndex] || null;
  };

  const addUploadedFiles = (files: File[]) => {
    const newItems: UploadedFileItem[] = files.map((f, idx) => {
      const ext = f.name.split('.').pop()?.toLowerCase();
      const isSar = ext === 'tif' || ext === 'tiff' || f.name.toLowerCase().includes('sar');
      const currentCount = uploadedFiles.length + idx;
      
      let defaultRole = 'Primary Image';
      if (currentCount === 0) defaultRole = isSar ? 'SAR Amplitude' : 'T0 / Baseline Image';
      else if (currentCount === 1) defaultRole = isSar ? 'SAR Amplitude' : 'T1 / Comparison Image';
      else defaultRole = `Image #${currentCount + 1}`;

      return {
        id: `upload-${Date.now()}-${idx}`,
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
      const filtered = prev.filter((f) => f.id !== id);
      if (filtered.length === 0) {
        setInputMode('preset');
      }
      return filtered;
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
    setInputMode('preset');
  };

  const executeAnalysis = async (customQueryOverride?: string) => {
    const activeQuery = customQueryOverride || query;
    if (!activeQuery.trim()) {
      setError('Query prompt cannot be empty.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setActiveArtifact(null);
    setActiveEvidence(null);

    setPipelineStages([
      { stage: 'Agent Intent Formulation & Routing', status: 'active' },
      { stage: 'Specialist Tool Pipeline Execution', status: 'pending' },
      { stage: 'Evidence Extraction & Artifact Generation', status: 'pending' },
      { stage: 'Vision-Language Synthesis', status: 'pending' },
    ]);

    try {
      let res: QueryResponse;

      if (inputMode === 'upload') {
        if (uploadedFiles.length === 0) {
          throw new Error('Please upload at least one satellite image file.');
        }

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
        selectedSceneIndex,
        selectSceneIndex,
        demoManifest: DEMO_MANIFEST,
        activeSceneSample: getActiveSceneSample(),
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
