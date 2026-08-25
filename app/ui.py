"""Presentation-ready Interactive Web Dashboard for SatQuery AI.

Provides the judging and demonstration interface directly from the FastAPI backend.
"""

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SatQuery AI — Agent Core & Orchestration Dashboard</title>
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
      background: rgba(18, 24, 38, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 16px 32px;
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
      font-size: 24px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
    }
    .brand-title {
      font-size: 18px;
      font-weight: 700;
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
    .badge-dev1 { background: rgba(59, 130, 246, 0.15); color: var(--accent-cyan); border-color: rgba(59, 130, 246, 0.3); }
    .badge-health { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.3); }
    
    main {
      flex: 1;
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 32px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* Specialist Registry Dashboard Bar */
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
      position: relative;
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
    .tool-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    .tool-tag {
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
    }

    /* Demo Presets Row */
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

    /* Interactive Query Console */
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

    /* Two-Column Results Grid */
    .results-layout {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
    }
    @media (max-width: 1024px) {
      .results-layout { grid-template-columns: 1fr; }
    }

    /* Decision & Plan Cards */
    .decision-card {
      background: rgba(59, 130, 246, 0.06);
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .decision-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .decision-label { color: var(--text-muted); }
    .decision-val { font-weight: 600; color: var(--text-main); }
    .why-box {
      margin-top: 10px;
      padding: 10px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.5;
      color: #93c5fd;
      border-left: 3px solid var(--accent-cyan);
    }

    /* Flowchart / Workflow Graph */
    .workflow-steps {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin: 12px 0;
    }
    .workflow-step {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      font-size: 13px;
    }
    .step-num {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: var(--accent-blue);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 700;
    }
    .step-info { flex: 1; }
    .step-title { font-weight: 600; }
    .step-desc { font-size: 11px; color: var(--text-muted); }
    .step-pill {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.2);
      color: var(--accent-emerald);
      font-weight: 600;
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
      padding: 3px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    .trace-time { color: var(--text-muted); font-size: 11px; }
    .trace-stage { color: var(--accent-cyan); font-weight: 600; }
    .trace-dur { color: var(--accent-amber); font-size: 11px; margin-left: auto; }

    /* Evidence List */
    .evidence-item {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      padding: 10px 14px;
      border-radius: 6px;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .evidence-head { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 4px; }
    .evidence-meta { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    .swap-toggle-btn {
      background: rgba(139, 92, 246, 0.15);
      color: #c4b5fd;
      border: 1px solid rgba(139, 92, 246, 0.3);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .swap-toggle-btn:hover { background: rgba(139, 92, 246, 0.3); }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span class="brand-icon">🛰️</span>
      <div>
        <div class="brand-title">SatQuery AI</div>
        <div style="font-size: 11px; color: var(--text-muted);">Division 1: Agent Core + Backend + Orchestration</div>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge badge-dev1">Lalith Praveen (Lead)</span>
      <span id="health-badge" class="badge badge-health">● Backend Ready</span>
      <button id="btn-swap-tool" class="swap-toggle-btn" onclick="toggleToolSwap()">⚡ Swap Specialist Mock (Demo)</button>
    </div>
  </header>

  <main>
    <!-- Specialist Registry Dashboard -->
    <div class="section-card">
      <div class="section-header">
        <div class="section-title">
          <span>🛠️ Active Specialist Tool Registry</span>
          <span style="font-size: 11px; font-weight: 400; color: var(--text-muted);">(Plug-and-play architecture for Divs 2, 3, 4)</span>
        </div>
      </div>
      <div id="registry-container" class="registry-grid">
        <!-- Dynamically loaded from /api/v1/tools -->
      </div>
    </div>

    <!-- Judge Demo Presets Row -->
    <div class="section-card">
      <div class="section-header">
        <div class="section-title">⚡ 1-Click Judge Presentation Presets</div>
        <span style="font-size: 12px; color: var(--text-muted);">Loads authentic remote-sensing rasters & executes live pipeline</span>
      </div>
      <div class="presets-container">
        <button class="preset-btn" onclick="loadDemo('A')">
          <span class="preset-title">🛰️ Demo A: Single-Image VQA</span>
          <span class="preset-desc">1 Optical Raster • Scene Classification & Aircraft Counting</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('B')">
          <span class="preset-title">🎯 Demo B: Grounding & Localization</span>
          <span class="preset-desc">1 Optical Raster • Spatial Runway Bounding Box Localization</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('C')">
          <span class="preset-title">⏳ Demo C: Bi-Temporal Change</span>
          <span class="preset-desc">2 Temporal Images (T0/T1) • Urban Expansion & Change Map</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('D')">
          <span class="preset-title">⚡ Demo D: Optical-SAR Fusion</span>
          <span class="preset-desc">Optical + SAR Pair • Cloud-Penetrating Radar Structure Fusion</span>
        </button>
        <button class="preset-btn" onclick="loadDemo('E')">
          <span class="preset-title">🔄 Demo E: Multi-Tool Sequential</span>
          <span class="preset-desc">2 Images • Change Detection ➔ Localized Characterization</span>
        </button>
      </div>

      <div class="query-box">
        <input type="text" id="query-input" class="query-input" placeholder="Type custom remote-sensing query or click a preset above..." value="What is the dominant land cover and infrastructure in this scene?">
        <button class="run-btn" onclick="submitCurrentQuery()">Execute Query</button>
      </div>
    </div>

    <!-- Main Results Grid -->
    <div class="results-layout">
      <!-- Left Column: Synthesized Answer, Evidence & Artifacts -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">💬 Synthesized Answer</div>
            <div id="confidence-badge" class="badge" style="background: rgba(59, 130, 246, 0.2); color: #93c5fd;">Confidence: 0.90</div>
          </div>
          <div id="answer-text" style="font-size: 15px; line-height: 1.6; color: #e5e7eb; min-height: 60px;">
            Click a preset or execute a query above to observe the live agent orchestration pipeline.
          </div>
        </div>

        <div class="section-card">
          <div class="section-header">
            <div class="section-title">🔍 Grounding Evidence & Visual Artifacts</div>
          </div>
          <div id="evidence-container">
            <div style="font-size: 13px; color: var(--text-muted);">No evidence generated yet. Run a query to inspect bounding boxes and difference maps.</div>
          </div>
          <div id="artifacts-container" style="margin-top: 12px;"></div>
        </div>

      </div>

      <!-- Right Column: Agent Decision, TaskPlan, & Live Audit Trace -->
      <div style="display: flex; flex-direction: column; gap: 20px;">

        <!-- Agent Decision Card -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">🧠 Agent Decision Card</div>
          </div>
          <div id="decision-content">
            <div class="decision-row"><span class="decision-label">Resolved Task:</span><span id="dec-task" class="decision-val">—</span></div>
            <div class="decision-row"><span class="decision-label">Input Count / Modalities:</span><span id="dec-modalities" class="decision-val">—</span></div>
            <div class="decision-row"><span class="decision-label">Selected Specialist:</span><span id="dec-specialist" class="decision-val">—</span></div>
            <div class="decision-row"><span class="decision-label">Workflow:</span><span id="dec-workflow" class="decision-val">—</span></div>
            <div class="why-box" id="dec-why">
              <strong>Why This Tool?</strong><br>
              <span id="dec-why-text">Select a query to view the operational routing rationale.</span>
            </div>
          </div>
        </div>

        <!-- Task Plan Visualizer -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">📋 Canonical Task Plan & Workflow</div>
          </div>
          <div id="taskplan-goal" style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">Goal: Awaiting query...</div>
          <div id="workflow-steps" class="workflow-steps">
            <!-- Dynamically populated from response.task_plan -->
          </div>
        </div>

        <!-- Execution Trace Timeline -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">⏱️ Operational Execution Trace</div>
            <span style="font-size: 11px; color: var(--text-muted);">Live Audit Trail</span>
          </div>
          <div id="trace-container" class="trace-timeline">
            <div class="trace-item"><span class="trace-time">00:00.000</span><span class="trace-stage">AWAITING_REQUEST</span></div>
          </div>
        </div>

      </div>
    </div>
  </main>

  <script>
    let currentDemoImages = [];
    let isSwappedMode = false;

    // Load registered tools on initial page load
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

    // Load 1-Click Judge Demo presets with authentic raster paths
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

      // Reset highlighted tool cards
      document.querySelectorAll('.tool-card').forEach(c => c.classList.remove('active-selected'));

      const payload = {
        query: query,
        images: currentDemoImages
      };

      try {
        const response = await fetch('/api/v1/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        renderResponse(data);
      } catch (err) {
        console.error("Query submission failed:", err);
      }
    }

    // Render backend results to UI components
    function renderResponse(data) {
      // 1. Answer & Confidence
      document.getElementById('answer-text').innerText = data.answer;
      document.getElementById('confidence-badge').innerText = data.confidence !== null ? `Confidence: ${(data.confidence * 100).toFixed(0)}%` : 'Confidence: N/A';

      // 2. Highlight selected specialist in registry
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
        document.getElementById('dec-modalities').innerText = `${dec.image_count} image(s) [${dec.detected_modalities.join(', ')}]`;
        document.getElementById('dec-specialist').innerText = dec.selected_specialist;
        document.getElementById('dec-workflow').innerText = dec.workflow_summary;
        document.getElementById('dec-why-text').innerText = dec.why_this_tool;
      }

      // 4. Task Plan & Workflow
      if (data.task_plan) {
        document.getElementById('taskplan-goal').innerText = `Goal: ${data.task_plan.goal}`;
        const wfContainer = document.getElementById('workflow-steps');
        wfContainer.innerHTML = '';
        data.task_plan.steps.forEach((step, idx) => {
          const stepEl = document.createElement('div');
          stepEl.className = 'workflow-step';
          stepEl.innerHTML = `
            <div class="step-num">${idx + 1}</div>
            <div class="step-info">
              <div class="step-title">${step.tool_name} (${step.task})</div>
              <div class="step-desc">${step.purpose}</div>
            </div>
            <span class="step-pill">${step.status.toUpperCase()}</span>
          `;
          wfContainer.appendChild(stepEl);
        });
      }

      // 5. Execution Trace
      const traceContainer = document.getElementById('trace-container');
      traceContainer.innerHTML = '';
      (data.execution_trace || []).forEach(tr => {
        const item = document.createElement('div');
        item.className = 'trace-item';
        const dur = tr.duration_ms ? `${tr.duration_ms}ms` : '';
        item.innerHTML = `
          <span class="trace-time">✓</span>
          <span class="trace-stage">${tr.stage}</span>
          <span style="color: var(--text-muted);">[${tr.component}]</span>
          <span class="trace-dur">${dur}</span>
        `;
        traceContainer.appendChild(item);
      });

      // 6. Evidence Items
      const evContainer = document.getElementById('evidence-container');
      evContainer.innerHTML = '';
      if (!data.evidence || data.evidence.length === 0) {
        evContainer.innerHTML = '<div style="font-size: 13px; color: var(--text-muted);">No localized bounding boxes in response.</div>';
      } else {
        data.evidence.forEach(ev => {
          const evEl = document.createElement('div');
          evEl.className = 'evidence-item';
          const conf = ev.confidence ? `Conf: ${(ev.confidence * 100).toFixed(0)}%` : '';
          evEl.innerHTML = `
            <div class="evidence-head">
              <span>📌 ${ev.label} (${ev.type})</span>
              <span style="color: var(--accent-emerald);">${conf}</span>
            </div>
            <div class="evidence-meta">Data: ${JSON.stringify(ev.data)}</div>
          `;
          evContainer.appendChild(evEl);
        });
      }

      // 7. Artifacts
      const artContainer = document.getElementById('artifacts-container');
      artContainer.innerHTML = '';
      (data.artifacts || []).forEach(art => {
        const artEl = document.createElement('div');
        artEl.className = 'badge';
        artEl.style.background = 'rgba(139, 92, 246, 0.2)';
        artEl.style.color = '#c4b5fd';
        artEl.innerHTML = `📁 Artifact: ${art.name} (${art.type})`;
        artContainer.appendChild(artEl);
      });
    }

    // Toggle tool swap live demo
    async function toggleToolSwap() {
      try {
        const res = await fetch('/api/v1/tools/swap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_tool: "single_image_vqa_mock", use_alternate: !isSwappedMode })
        });
        const result = await res.json();
        isSwappedMode = !isSwappedMode;
        document.getElementById('btn-swap-tool').innerText = isSwappedMode ? "⚡ Active: Swapped Tool (Demo v2.0)" : "⚡ Swap Specialist Mock (Demo)";
        await loadRegistry();
        alert(`Registry updated: ${result.message}`);
      } catch (err) {
        alert("Tool swap error: " + err);
      }
    }

    // Initialize registry on page load
    window.addEventListener('DOMContentLoaded', () => {
      loadRegistry();
      loadDemo('A');
    });
  </script>
</body>
</html>
"""
