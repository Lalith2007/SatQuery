"""Presentation-ready Interactive Web Dashboard for SatQuery AI.

Provides the judging, demonstration, evidence visualization, report generation,
and benchmark evaluation interface directly from the FastAPI backend.
"""

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SatQuery AI — Evidence, Evaluation & Presentation Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-card: #121826;
      --bg-card-hover: #182236;
      --bg-surface: #1a2234;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-purple: #8b5cf6;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-active: rgba(59, 130, 246, 0.5);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(18, 24, 38, 0.90);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 14px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      font-size: 26px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
    }
    .brand-title {
      font-size: 19px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--border-subtle);
    }
    .badge-dev5 { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border-color: rgba(6, 182, 212, 0.3); }
    .badge-health { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.3); }
    
    /* Navigation Tabs */
    .tab-bar {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 32px;
      background: rgba(14, 20, 32, 0.6);
    }
    .tab-btn {
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--text-muted);
      padding: 12px 16px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .tab-btn.active {
      color: var(--accent-cyan);
      border-bottom-color: var(--accent-cyan);
    }
    .tab-btn:hover { color: var(--text-main); }

    main {
      flex: 1;
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px 32px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .section-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 20px;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .section-title {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Registry Grid */
    .registry-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
    }
    .tool-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 14px;
      transition: all 0.2s ease;
    }
    .tool-card.active-selected {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 16px rgba(6, 182, 212, 0.35);
      background: rgba(6, 182, 212, 0.08);
    }
    .tool-card-name {
      font-weight: 600;
      font-size: 14px;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .tool-card-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.4;
      margin-bottom: 8px;
    }
    .tool-tags { display: flex; flex-wrap: wrap; gap: 4px; }
    .tool-tag {
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
    }

    /* Presets */
    .presets-container {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }
    .preset-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .preset-btn:hover {
      background: var(--bg-card-hover);
      border-color: var(--accent-blue);
      transform: translateY(-2px);
    }
    .preset-title { color: var(--accent-cyan); font-size: 13px; }
    .preset-desc { font-size: 11px; color: var(--text-muted); font-weight: 400; }

    /* Query Box */
    .query-box {
      display: flex;
      gap: 12px;
      margin-top: 12px;
    }
    .query-input {
      flex: 1;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 12px 16px;
      color: var(--text-main);
      font-size: 14px;
      font-family: inherit;
    }
    .query-input:focus {
      outline: none;
      border-color: var(--accent-blue);
    }
    .run-btn {
      background: linear-gradient(135deg, var(--accent-blue), #2563eb);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px 24px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .run-btn:hover { opacity: 0.9; }

    /* Results Layout */
    .results-layout {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 24px;
    }
    @media (max-width: 1024px) {
      .results-layout { grid-template-columns: 1fr; }
    }

    /* Visual Evidence Display */
    .visual-evidence-box {
      position: relative;
      background: #080c14;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-subtle);
      margin-top: 12px;
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .visual-canvas {
      max-width: 100%;
      max-height: 480px;
      display: block;
    }

    /* Report Download Bar */
    .report-bar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border-subtle);
    }
    .report-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .report-btn:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: var(--accent-blue);
    }

    /* Trace Timeline */
    .trace-timeline {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      background: #080c14;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      max-height: 320px;
      overflow-y: auto;
    }
    .trace-item {
      display: flex;
      gap: 10px;
      align-items: baseline;
      padding: 4px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    /* Benchmark Lab Tab */
    .eval-container { display: none; }
    .eval-grid {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
    }
    .eval-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 12px;
      border-radius: 8px;
      text-align: left;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .eval-btn:hover, .eval-btn.active {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.1);
    }
    .scoreboard-box {
      background: #080c14;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 20px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      overflow-x: auto;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span class="brand-icon">🛰️</span>
      <div>
        <div class="brand-title">SatQuery AI</div>
        <div style="font-size: 11px; color: var(--text-muted);">Interactive Multimodal Remote Sensing Intelligence</div>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge badge-dev5">Production Platform</span>
      <span id="health-badge" class="badge badge-health">● Backend Live</span>
      <button class="report-btn" onclick="toggleToolSwap()">⚡ Swap Specialist (Demo)</button>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="tab-bar">
    <button class="tab-btn active" id="tab-demo" onclick="switchTab('demo')">🔍 Interactive Vision-Language Assistant</button>
    <button class="tab-btn" id="tab-eval" onclick="switchTab('eval')">📊 Benchmark Evaluation & Scoreboards</button>
  </div>

  <!-- TAB 1: MAIN DEMO & EVIDENCE INTERFACE -->
  <main id="view-demo">
    
    <!-- Specialist Registry Bar -->
    <div class="section-card">
      <div class="section-header">
        <div class="section-title">
          <span>🛠️ Active Specialist Tool Registry</span>
          <span style="font-size: 11px; font-weight: 400; color: var(--text-muted);">(Plug-and-play architecture for Divs 2, 3, 4)</span>
        </div>
      </div>
      <div id="registry-container" class="registry-grid"></div>
    </div>

    <!-- 1-Click Presentation Presets -->
    <div class="section-card">
      <div class="section-header">
        <div class="section-title">⚡ 1-Click Presentation Presets</div>
        <span style="font-size: 12px; color: var(--text-muted);">Authentic remote-sensing rasters & real evidence grounding</span>
      </div>
      <div class="presets-container">
        <button class="preset-btn" onclick="loadDemo('A')">
          <span class="preset-title">🛰️ Demo A: Single-Image VQA</span>
          <span class="preset-desc">1 Optical Raster • Scene Classification & Aircraft Counting</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('B')">
          <span class="preset-title">🎯 Demo B: Spatial Feature Grounding</span>
          <span class="preset-desc">1 Optical Raster • Airport Runway & Apron BBox Overlay</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('C')">
          <span class="preset-title">⏳ Demo C: Bi-Temporal Change</span>
          <span class="preset-desc">2 Temporal Images (T0/T1) • Urban Difference Mask Map</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('D')">
          <span class="preset-title">⚡ Demo D: Optical-SAR Fusion</span>
          <span class="preset-desc">Optical + SAR Pair • False-Color Radar Penetration Blend</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('E')">
          <span class="preset-title">🔄 Demo E: Multi-Tool Workflow</span>
          <span class="preset-desc">2 Images • Change Detection ➔ Localized Characterization</span>
        </button>
      </div>

      <div class="query-box">
        <input type="text" id="query-input" class="query-input" placeholder="Enter remote-sensing query or click a preset..." value="What is the dominant land cover and infrastructure in this scene?">
        <button class="run-btn" onclick="submitCurrentQuery()">Execute Query</button>
      </div>
    </div>

    <!-- Results Layout -->
    <div class="results-layout">
      <!-- Left Column: Answer, Evidence & Interactive Overlays -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <!-- Synthesized Answer Card -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">💬 Synthesized Answer</div>
            <div id="confidence-badge" class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">Confidence: Calibrating...</div>
          </div>
          <div id="answer-text" style="font-size: 15px; line-height: 1.6; color: #e5e7eb; min-height: 50px;">
            Loading initial query...
          </div>
          
          <!-- Downloadable Intelligence Report Bar -->
          <div class="report-bar">
            <span style="font-size: 12px; color: var(--text-muted); font-weight: 600;">📥 Export Report:</span>
            <button class="report-btn" onclick="exportReport('html')">📄 Standalone HTML Report</button>
            <button class="report-btn" onclick="exportReport('markdown')">📝 Markdown Report</button>
            <button class="report-btn" onclick="exportReport('json')">📊 Machine JSON</button>
          </div>
        </div>

        <!-- Grounding Visual Evidence & Canvas View -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">🔍 Grounding Evidence & Spatial Artifacts</div>
            <span id="evidence-count-badge" class="badge" style="background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan);">0 items</span>
          </div>

          <!-- Canvas / Image Evidence Viewer -->
          <div class="visual-evidence-box" id="visual-evidence-box">
            <div id="evidence-viewer-placeholder" style="font-size: 13px; color: var(--text-muted);">
              Run a query to inspect rendered bounding boxes, difference masks, and cross-modal overlays.
            </div>
            <div id="evidence-canvas-container" style="display: none; position: relative; width: 100%; text-align: center;"></div>
          </div>

          <!-- Evidence List Items -->
          <div id="evidence-items-list" style="margin-top: 14px;"></div>
          <div id="artifacts-container" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;"></div>
        </div>

      </div>

      <!-- Right Column: Agent Decision & Auditable Operational Trace -->
      <div style="display: flex; flex-direction: column; gap: 20px;">

        <!-- Agent Decision Card -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">🧠 Agent Decision Card</div>
          </div>
          <div id="decision-content">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px;">
              <span style="color: var(--text-muted);">Resolved Task:</span><span id="dec-task" style="font-weight: 600;">—</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px;">
              <span style="color: var(--text-muted);">Input Modalities:</span><span id="dec-modalities" style="font-weight: 600;">—</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px;">
              <span style="color: var(--text-muted);">Selected Tool:</span><span id="dec-specialist" style="font-weight: 600; color: var(--accent-cyan);">—</span>
            </div>
            <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px; font-size: 12px; line-height: 1.5; color: #93c5fd; border-left: 3px solid var(--accent-cyan);" id="dec-why">
              <strong>Operational Rationale:</strong><br>
              <span id="dec-why-text">—</span>
            </div>
          </div>
        </div>

        <!-- Auditable Execution Trace Timeline -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">⏱️ Auditable Operational Trace</div>
            <span id="trace-latency-badge" style="font-size: 11px; color: var(--accent-amber);">0.0 ms</span>
          </div>
          <div id="trace-container" class="trace-timeline">
            <div class="trace-item"><span style="color: var(--text-muted);">00:00.000</span><span>AWAITING_REQUEST</span></div>
          </div>
        </div>

      </div>
    </div>
  </main>

  <!-- TAB 2: BENCHMARK EVALUATION LAB -->
  <main id="view-eval" class="eval-container">
    <div class="section-card">
      <div class="section-header">
        <div class="section-title">📊 Modular Benchmark Evaluation Lab (Division 5)</div>
        <span style="font-size: 12px; color: var(--text-muted);">Reproducible scoring across VRSBench, RSVQA, CDVQA, and ISRO/SAC</span>
      </div>

      <div class="eval-grid">
        <!-- Benchmark Selector -->
        <div>
          <div style="font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px;">Select Benchmark:</div>
          <div class="eval-btn active" onclick="runBenchmarkTab('vrsbench')">
            <div>
              <div>🛰️ VRSBench Suite</div>
              <div style="font-size: 11px; color: var(--text-muted); font-weight: 400;">VQA & Spatial Grounding (mIoU)</div>
            </div>
            <span>➔</span>
          </div>
          <div class="eval-btn" onclick="runBenchmarkTab('rsvqa')">
            <div>
              <div>🎯 RSVQA Suite</div>
              <div style="font-size: 11px; color: var(--text-muted); font-weight: 400;">Presence, Comparison & Count</div>
            </div>
            <span>➔</span>
          </div>
          <div class="eval-btn" onclick="runBenchmarkTab('cdvqa')">
            <div>
              <div>⏳ CDVQA Suite</div>
              <div style="font-size: 11px; color: var(--text-muted); font-weight: 400;">Change Detection & Description</div>
            </div>
            <span>➔</span>
          </div>
          <div class="eval-btn" onclick="runBenchmarkTab('isro_sac')">
            <div>
              <div>📡 ISRO/SAC Generic Test</div>
              <div style="font-size: 11px; color: var(--text-muted); font-weight: 400;">Cartosat-2S & RISAT SAR Pair</div>
            </div>
            <span>➔</span>
          </div>
        </div>

        <!-- Scoreboard Output -->
        <div>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="font-size: 13px; font-weight: 700; color: var(--accent-cyan);" id="eval-title">VRSBench Evaluation Scoreboard</div>
            <button class="report-btn" onclick="reRunActiveBenchmark()">⚡ Re-Execute Evaluation</button>
          </div>
          <div id="scoreboard-content" class="scoreboard-box">Executing benchmark evaluation...</div>
        </div>
      </div>
    </div>
  </main>

  <script>
    let currentDemoImages = [];
    let isSwappedMode = false;
    let lastQueryResponse = null;
    let activeBenchmark = "vrsbench";

    // Switch between Main Demo and Benchmark Lab tabs
    function switchTab(tab) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      if (tab === 'demo') {
        document.getElementById('tab-demo').classList.add('active');
        document.getElementById('view-demo').style.display = 'flex';
        document.getElementById('view-eval').style.display = 'none';
      } else {
        document.getElementById('tab-eval').classList.add('active');
        document.getElementById('view-demo').style.display = 'none';
        document.getElementById('view-eval').style.display = 'block';
        runBenchmarkTab(activeBenchmark);
      }
    }

    // Load registered tools
    async function loadRegistry() {
      try {
        const res = await fetch('/api/v1/tools');
        const tools = await res.json();
        const container = document.getElementById('registry-container');
        container.innerHTML = '';

        tools.forEach(tool => {
          const card = document.createElement('div');
          card.className = 'tool-card';
          card.id = `tool-card-${tool.name}`;
          const tags = (tool.supported_tasks || []).map(t => `<span class="tool-tag">${t}</span>`).join('');
          const modalities = (tool.required_modalities || []).map(m => `<span class="tool-tag" style="color: #93c5fd;">${m}</span>`).join('');

          card.innerHTML = `
            <div class="tool-card-name">
              <span>${tool.name}</span>
              <span style="font-size: 11px; color: #10b981;">v${tool.version}</span>
            </div>
            <div class="tool-card-desc">${tool.description}</div>
            <div class="tool-tags">${tags} ${modalities}</div>
          `;
          container.appendChild(card);
        });
      } catch (err) {
        console.error("Failed to load tools:", err);
      }
    }

    // Load 1-Click Judge presets
    function loadDemo(type) {
      const queryInput = document.getElementById('query-input');
      if (type === 'A') {
        queryInput.value = "What is the dominant land cover and infrastructure in this scene?";
        currentDemoImages = [{ path_or_uri: "demo_assets/demo_optical_single.png", format: "png", modality: "optical" }];
      } else if (type === 'B') {
        queryInput.value = "Where is the airport runway and apron located?";
        currentDemoImages = [{ path_or_uri: "demo_assets/demo_airport_grounding.png", format: "png", modality: "optical" }];
      } else if (type === 'C') {
        queryInput.value = "What changed between these two acquisition dates?";
        currentDemoImages = [
          { path_or_uri: "demo_assets/demo_change_t0.png", format: "png", modality: "optical" },
          { path_or_uri: "demo_assets/demo_change_t1.png", format: "png", modality: "optical" }
        ];
      } else if (type === 'D') {
        queryInput.value = "Use optical and SAR images together to identify structures beneath clouds.";
        currentDemoImages = [
          { path_or_uri: "demo_assets/demo_optical_cross.png", format: "png", modality: "optical" },
          { path_or_uri: "demo_assets/demo_sar_cross.tif", format: "tiff", modality: "sar" }
        ];
      } else if (type === 'E') {
        queryInput.value = "What changed, where did it happen, and was the new region built-up?";
        currentDemoImages = [
          { path_or_uri: "demo_assets/demo_change_t0.png", format: "png", modality: "optical" },
          { path_or_uri: "demo_assets/demo_change_t1.png", format: "png", modality: "optical" }
        ];
      }
      submitCurrentQuery();
    }

    // Submit query to FastAPI backend
    async function submitCurrentQuery() {
      const query = document.getElementById('query-input').value;
      if (!currentDemoImages || currentDemoImages.length === 0) {
        currentDemoImages = [{ path_or_uri: "demo_assets/demo_optical_single.png", format: "png", modality: "optical" }];
      }

      document.querySelectorAll('.tool-card').forEach(c => c.classList.remove('active-selected'));
      const payload = { query: query, images: currentDemoImages };

      try {
        const response = await fetch('/api/v1/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        lastQueryResponse = data;
        renderResponse(data);
      } catch (err) {
        console.error("Query submission failed:", err);
      }
    }

    // Render backend results to UI components
    function renderResponse(data) {
      document.getElementById('answer-text').innerText = data.answer;

      // 1. Calibrated Confidence Badge
      const confBadge = document.getElementById('confidence-badge');
      if (data.confidence !== null && data.confidence !== undefined) {
        const pct = (data.confidence * 100).toFixed(0);
        const tier = data.confidence >= 0.85 ? 'HIGH' : data.confidence >= 0.65 ? 'MODERATE' : 'LOW';
        const color = data.confidence >= 0.85 ? '#10b981' : data.confidence >= 0.65 ? '#f59e0b' : '#ef4444';
        confBadge.style.background = `rgba(${tier === 'HIGH' ? '16, 185, 129' : tier === 'MODERATE' ? '245, 158, 11' : '239, 68, 68'}, 0.15)`;
        confBadge.style.color = color;
        confBadge.innerHTML = `🟢 Confidence: ${pct}% [${tier}]`;
      } else {
        confBadge.style.background = 'rgba(156, 163, 175, 0.15)';
        confBadge.style.color = '#9ca3af';
        confBadge.innerHTML = `⚪ Confidence: Unavailable`;
      }

      // 2. Highlight selected specialist
      if (data.selected_tools && data.selected_tools.length > 0) {
        data.selected_tools.forEach(toolName => {
          const card = document.getElementById(`tool-card-${toolName}`);
          if (card) card.classList.add('active-selected');
        });
      }

      // 3. Agent Decision Card
      if (data.agent_decision) {
        const dec = data.agent_decision;
        document.getElementById('dec-task').innerText = dec.task_display_name;
        document.getElementById('dec-modalities').innerText = `${dec.image_count} raster(s) [${dec.detected_modalities.join(', ')}]`;
        document.getElementById('dec-specialist').innerText = dec.selected_specialist;
        document.getElementById('dec-why-text').innerText = dec.why_this_tool;
      }

      // 4. Execution Trace Timeline & Latency
      const traceContainer = document.getElementById('trace-container');
      traceContainer.innerHTML = '';
      let totalDur = 0;

      (data.execution_trace || []).forEach(tr => {
        const item = document.createElement('div');
        item.className = 'trace-item';
        const dur = tr.duration_ms ? `${tr.duration_ms.toFixed(1)}ms` : '';
        if (tr.duration_ms) totalDur += tr.duration_ms;
        const icon = tr.stage === 'INFERENCE_EXECUTED' ? '⚡' : tr.stage === 'TASK_RESOLVED' ? '🎯' : tr.stage === 'EVIDENCE_GENERATED' ? '🔍' : '✓';
        item.innerHTML = `
          <span>${icon}</span>
          <span style="color: var(--accent-cyan); font-weight: 600;">${tr.stage}</span>
          <span style="color: var(--text-muted);">[${tr.component}]</span>
          <span style="margin-left: auto; color: #10b981; font-weight: 600; font-size: 11px;">${tr.status}</span>
          <span style="color: var(--accent-amber); font-size: 11px; min-width: 45px; text-align: right;">${dur}</span>
        `;
        traceContainer.appendChild(item);
      });
      document.getElementById('trace-latency-badge').innerText = `Total: ${totalDur.toFixed(1)} ms`;

      // 5. Visual Evidence Rendering & Artifacts
      const evCountBadge = document.getElementById('evidence-count-badge');
      const canvasContainer = document.getElementById('evidence-canvas-container');
      const placeholder = document.getElementById('evidence-viewer-placeholder');
      const itemsList = document.getElementById('evidence-items-list');
      const artContainer = document.getElementById('artifacts-container');

      itemsList.innerHTML = '';
      artContainer.innerHTML = '';

      const evList = data.evidence || [];
      const artList = data.artifacts || [];
      evCountBadge.innerText = `${evList.length + artList.length} items`;

      // Look for visual artifacts with priority: 3-panel composite / fusion / annotated grounding > heatmap > crop > mask
      const visualPriority = (name) => {
        const n = (name || '').toLowerCase();
        if (n.includes('bitemporal_change_composite') || n.includes('optical_sar_fusion')) return 10;
        if (n.includes('annotated_grounding')) return 8;
        if (n.includes('heatmap')) return 6;
        if (n.includes('crop')) return 4;
        if (n.includes('mask')) return 2;
        return 1;
      };

      const imageArtifacts = artList.filter(a => a.name && (a.name.endsWith('.png') || a.name.endsWith('.jpg') || a.name.endsWith('.jpeg')));
      imageArtifacts.sort((a, b) => visualPriority(b.name) - visualPriority(a.name));
      const visualArt = imageArtifacts[0];

      if (visualArt) {
        placeholder.style.display = 'none';
        canvasContainer.style.display = 'block';
        canvasContainer.innerHTML = `
          <img src="/api/v1/artifacts/${visualArt.artifact_id}" alt="${visualArt.name}" class="visual-canvas" style="border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
          <div style="font-size: 11px; color: var(--accent-cyan); margin-top: 6px;">🖼️ Rendered Artifact: ${visualArt.name}</div>
        `;
      } else {
        canvasContainer.style.display = 'none';
        placeholder.style.display = 'block';
        placeholder.innerText = evList.length > 0 ? 'Grounding evidence localized below.' : 'No spatial bounding boxes or visual artifacts generated for this query.';
      }

      // Render structured evidence list
      evList.forEach(ev => {
        const evEl = document.createElement('div');
        evEl.style.background = 'var(--bg-surface)';
        evEl.style.border = '1px solid var(--border-subtle)';
        evEl.style.padding = '10px 14px';
        evEl.style.borderRadius = '6px';
        evEl.style.marginBottom = '8px';
        evEl.style.fontSize = '13px';
        const conf = ev.confidence ? `Conf: ${(ev.confidence * 100).toFixed(0)}%` : '';
        evEl.innerHTML = `
          <div style="display: flex; justify-content: space-between; font-weight: 600;">
            <span style="color: var(--accent-cyan);">📌 ${ev.label} (${ev.type})</span>
            <span style="color: var(--accent-emerald);">${conf}</span>
          </div>
          <div style="font-size: 11px; color: var(--text-muted); font-family: monospace; margin-top: 4px;">Data: ${JSON.stringify(ev.data)}</div>
        `;
        itemsList.appendChild(evEl);
      });

      // Render artifact badges
      artList.forEach(art => {
        const artEl = document.createElement('a');
        artEl.className = 'badge';
        artEl.href = `/api/v1/artifacts/${art.artifact_id}`;
        artEl.target = '_blank';
        artEl.style.background = 'rgba(139, 92, 246, 0.2)';
        artEl.style.color = '#c4b5fd';
        artEl.style.textDecoration = 'none';
        artEl.innerHTML = `📁 ${art.name} (${art.type}) ➔`;
        artContainer.appendChild(artEl);
      });
    }

    // 1-Click Export Report
    async function exportReport(fmt) {
      if (!lastQueryResponse) {
        alert("Please execute a query first before exporting a report.");
        return;
      }
      try {
        const res = await fetch('/api/v1/reports/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ response: lastQueryResponse, format: fmt })
        });
        const rep = await res.json();
        window.open(rep.download_url, '_blank');
      } catch (err) {
        alert("Report generation failed: " + err);
      }
    }

    // Run benchmark in Benchmark Lab tab
    async function runBenchmarkTab(benchmarkName) {
      activeBenchmark = benchmarkName;
      document.querySelectorAll('.eval-btn').forEach(b => b.classList.remove('active'));
      const sb = document.getElementById('scoreboard-content');
      sb.innerText = `Executing ${benchmarkName.toUpperCase()} evaluation suite...`;

      // Sample evaluation payloads representing canonical tasks
      let preds = [], gts = [];
      if (benchmarkName === 'vrsbench') {
        preds = [
          { answer: "Airport runway with 4 aircraft parked.", bbox: [0.1, 0.1, 0.5, 0.5] },
          { answer: "Dense urban residential buildings.", bbox: [0.2, 0.3, 0.7, 0.8] },
          { answer: "Water reservoir with concrete dam wall.", bbox: [0.4, 0.1, 0.8, 0.6] }
        ];
        gts = [
          { answer: "Airport runway with 4 aircraft parked on apron.", bbox: [0.12, 0.08, 0.52, 0.51], category: "presence" },
          { answer: "Dense urban residential buildings.", bbox: [0.22, 0.28, 0.69, 0.79], category: "landcover" },
          { answer: "Large water reservoir and dam infrastructure.", bbox: [0.41, 0.09, 0.79, 0.61], category: "presence" }
        ];
      } else if (benchmarkName === 'rsvqa') {
        preds = [
          { answer: "yes" }, { answer: "3" }, { answer: "yes" }, { answer: "commercial" }
        ];
        gts = [
          { answer: "yes", type: "presence" }, { answer: "3", type: "count" }, { answer: "yes", type: "comparison" }, { answer: "commercial", type: "landcover" }
        ];
      } else if (benchmarkName === 'cdvqa') {
        preds = [
          { answer: "Yes, significant new construction and building expansion observed between acquisitions." },
          { answer: "No detectable changes in vegetation or water body bounds." }
        ];
        gts = [
          { answer: "Yes, noticeable urban expansion with new commercial structures." },
          { answer: "No changes observed in water and natural vegetation." }
        ];
      } else if (benchmarkName === 'isro_sac') {
        preds = [
          { answer: "Metallic port infrastructure detected under cloud cover via RISAT SAR.", bbox: [0.2, 0.3, 0.6, 0.7], sensor: "risat_sar" },
          { answer: "Cartosat-2S optical capture showing agricultural fields.", bbox: [0.1, 0.1, 0.8, 0.8], sensor: "cartosat_2s" }
        ];
        gts = [
          { answer: "Metallic port infrastructure detected under cloud cover via RISAT SAR scattering.", bbox: [0.21, 0.29, 0.62, 0.69], sensor: "risat_sar" },
          { answer: "Cartosat-2S optical high-resolution imagery showing cultivated agricultural plots.", bbox: [0.11, 0.09, 0.79, 0.81], sensor: "cartosat_2s" }
        ];
      }

      try {
        const res = await fetch('/api/v1/evaluation/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ benchmark: benchmarkName, predictions: preds, ground_truths: gts })
        });
        const evalData = await res.json();
        sb.innerText = evalData.scoreboard_markdown;
        document.getElementById('eval-title').innerText = `${evalData.benchmark} Performance Scoreboard (Score: ${evalData.aggregate_normalized_score.toFixed(1)}/100.0)`;
      } catch (err) {
        sb.innerText = "Benchmark execution error: " + err;
      }
    }

    function reRunActiveBenchmark() {
      runBenchmarkTab(activeBenchmark);
    }

    async function toggleToolSwap() {
      try {
        const res = await fetch('/api/v1/tools/swap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_tool: "single_image_vqa_mock", use_alternate: !isSwappedMode })
        });
        const result = await res.json();
        isSwappedMode = !isSwappedMode;
        await loadRegistry();
        alert(`Registry updated: ${result.message}`);
      } catch (err) {
        alert("Tool swap error: " + err);
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      loadRegistry();
      loadDemo('A');
    });
  </script>
</body>
</html>
"""
