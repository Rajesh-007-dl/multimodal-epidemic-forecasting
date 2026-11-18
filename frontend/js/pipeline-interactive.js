/**
 * pipeline-interactive.js
 * Interactive 5-Stage Scientific Architecture Pipeline & Advanced Modal Zoom Controller
 * Illuminates data flow strokes, updates explanatory panel, and loads diagram previews
 * Full typographic and mathematical symbol formatting (Delta, beta, gamma, sigma, lambda, R^2)
 */

const PIPELINE_STAGES = {
  phase1: {
    title: "Phase I: Multimodal Data Ingestion",
    tagline: "Synchronizing Global Epidemic Dynamics with Societal Mobility & Policy",
    badge: "1,143 Consecutive Global Days",
    diagram: "assets/images/paper_fig1_system_framework.png",
    diagramCaption: "Figure 1: Multimodal Epidemic Forecasting Framework Architecture",
    description: "Ingests daily global infection statistics alongside multi-sector human mobility and government non-pharmaceutical interventions (NPIs) to form a unified spatio-temporal tensor.",
    specs: [
      { label: "Epidemiological Data", value: "JHU CSSE (Confirmed, Deaths, Recoveries)" },
      { label: "Human Mobility", value: "Google Mobility Reports (6 Societal Sectors)" },
      { label: "Policy Interventions", value: "Oxford Blavatnik Stringency Index (OxCGRT)" },
      { label: "Tensor Dimensions", value: "Batch B × Sequence 14–56 × 10 Channels" }
    ],
    mathLabel: "Spatio-Temporal Ingestion Tensor",
    formula: "𝐗<sub>t</sub> = [ Y<sub>conf</sub>(t), Y<sub>death</sub>(t), Y<sub>rec</sub>(t), S<sub>policy</sub>(t), M₁(t), …, M₆(t) ]ᵀ",
    mathLatex: "\\mathbf{X}_t = \\left[ Y_{\\text{conf}}(t), Y_{\\text{death}}(t), Y_{\\text{rec}}(t), S_{\\text{policy}}(t), M_1(t), \\dots, M_6(t) \\right]^T"
  },
  phase2: {
    title: "Phase II: Chronological Partitioning & Preprocessing",
    tagline: "Rigorous Temporal Alignment with Outlier Filtering & Differencing",
    badge: "Leak-Free Time Horizon Splitting",
    diagram: "assets/images/paper_fig1_system_framework.png",
    diagramCaption: "Figure 1: Chronological Partitioning & Imputation Pipeline",
    description: "Applies first-order daily differencing to stabilize non-stationary trends, followed by rolling median imputation to mitigate retrospective weekend reporting lags. Partitions data chronologically to strictly prohibit lookahead data leakage.",
    specs: [
      { label: "Statistical Horizon", value: "Days 1–66 (Train) | Days 67–76 (10-Day Holdout)" },
      { label: "Deep Learning Horizon", value: "Jan 2020 – Mar 2023 (80% Train | 20% Held-Out)" },
      { label: "Stationarity Transform", value: "ΔY(t) = Y(t) − Y(t−1)  (first difference, d = 1)" },
      { label: "Data Integrity", value: "Strict zero-lookahead chronological split" }
    ],
    mathLabel: "Stationarity Differencing Operator",
    formula: "ΔY<sub>t</sub> = (1 − B)<sup>d</sup> Y<sub>t</sub> &nbsp;&nbsp;(where B is the backshift lag operator, d = 1)",
    mathLatex: "\\Delta Y_t = (1 - B)^d Y_t, \\quad d = 1"
  },
  phase3: {
    title: "Phase III: Multi-Paradigm Modeling Zoo",
    tagline: "Four Architectural Approaches: From Statistical Hybrids to PINNs and Transformers",
    badge: "4 Evaluated Paradigms",
    diagram: "assets/images/paper_fig1d_neural_seir_pinn.png",
    diagramCaption: "Figure 4: Physics-Informed Epidemic Neural Network (Neural SEIR-PINN)",
    description: "Integrates four modeling philosophies: Linear Autoregressive XGBoost, Exponential Smoothing Random Forest, Physics-Informed Neural Networks, and Channel-Independent Patch Transformers.",
    submodels: [
      { id: "pinn", name: "Neural-SEIR-PINN", diagram: "assets/images/paper_fig1d_neural_seir_pinn.png", desc: "Bi-LSTM encoder coupled with compartmental SEIR differential equations enforcing conservation laws." },
      { id: "patchtst", name: "PatchTST Transformer", diagram: "assets/images/paper_fig1e_patchtst_transformer.png", desc: "Subseries patching (P=8, S=4) preserving local temporal context with multi-head self-attention." },
      { id: "arima_xgb", name: "Hybrid ARIMA-XGBoost", diagram: "assets/images/paper_fig1b_hybrid_arima_xgboost.png", desc: "Linear parametric Auto-ARIMA baseline with lag-7 non-linear residual boosting." },
      { id: "hw_rf", name: "Hybrid HW-Random Forest", diagram: "assets/images/paper_fig1c_hybrid_hw_rf.png", desc: "Holt-Winters level/trend smoothing combined with a 100-tree bagged residual forest." }
    ],
    specs: [
      { label: "Neural-SEIR-PINN", value: "Bi-LSTM + Dynamic β(t), γ(t) Parameter Heads" },
      { label: "PatchTST Structure", value: "3-Layer Transformer (8 Heads, d<sub>model</sub> = 128)" },
      { label: "Loss Function", value: "ℒ<sub>total</sub> = ℒ<sub>data</sub> + λ<sub>physics</sub> · ℒ<sub>SEIR</sub>" },
      { label: "Regularization", value: "λ = 0.05 (AdamW Optimizer, lr = 1e-3)" }
    ],
    mathLabel: "SEIR ODE Invariant & Reproduction Ratio",
    formula: "dI/dt = σE − γ(t)I &nbsp;&nbsp;|&nbsp;&nbsp; R<sub>t</sub> = β(t) / γ(t) &nbsp;&nbsp;|&nbsp;&nbsp; d(S+E+I+R)/dt = 0",
    mathLatex: "\\frac{dI}{dt} = \\sigma E - \\gamma(t) I, \\quad R_t = \\frac{\\beta(t)}{\\gamma(t)}"
  },
  phase4: {
    title: "Phase IV: Out-of-Sample Forecasting",
    tagline: "Direct Multi-Horizon Rollouts Without Autoregressive Error Cascades",
    badge: "10 to 15-Day Multi-Horizon",
    diagram: "assets/images/paper_fig8_merged_confirmed_pinn_and_patchtst.png",
    diagramCaption: "Figure 8: Comparative Confirmed Cases Forecasts (PINN vs PatchTST)",
    description: "Evaluates multi-step forecasting stability under out-of-sample conditions. PatchTST utilizes direct projection heads to eliminate recursive multi-step compounding drift, while Neural-SEIR-PINN guides trajectories along biologically feasible manifolds.",
    specs: [
      { label: "Forecast Horizon", value: "10-Day Replication & 15-Day Multi-Step Head" },
      { label: "Error Compounding", value: "Direct projection eliminates autoregressive drift" },
      { label: "Physical Validity", value: "Conservation constraints enforce non-negative counts" },
      { label: "Target Series", value: "Confirmed Infections, Cumulative Recoveries, Deaths" }
    ],
    mathLabel: "Direct Multi-Horizon Transformer Projection",
    formula: "Ŷ<sub>t+1:t+H</sub> = 𝐖<sub>proj</sub> · TransformerEncoder(𝐏<sub>t</sub>)",
    mathLatex: "\\hat{\\mathbf{Y}}_{t+1:t+H} = \\mathbf{W}_{\\text{proj}} \\cdot \\text{TransformerEncoder}(\\mathbf{P}_t)"
  },
  phase5: {
    title: "Phase V: Validation & Uncertainty Engine",
    tagline: "Statistical Goodness-of-Fit with Parametric Confidence Bounds",
    badge: "p < 0.001 Statistical Significance",
    diagram: "assets/images/paper_table_comparison_sota.png",
    diagramCaption: "Table I: Statistical Performance Comparison with Related Benchmark Studies",
    description: "Rigorous empirical holdout evaluation against isolated ground truth. Generates 80%, 90%, and 95% parametric confidence intervals using Student's t-distributions adjusted for residual variance across 8 benchmark literature studies and our 4 proposed models.",
    specs: [
      { label: "Peak Recovery Fit", value: "R² = 0.976 (PatchTST), R² = 0.972 (PINN)" },
      { label: "Confidence Intervals", value: "80%, 90%, 95% Student's t-Bounds (df = n − 2)" },
      { label: "Baseline Improvement", value: "94.8% Reduction in cumulative multi-step error" },
      { label: "Metrics Evaluated", value: "R², Adjusted R², MAE, MSE, RMSE, MAPE" }
    ],
    mathLabel: "Parametric Student's t Prediction Interval",
    formula: "Prediction Interval = Ŷ<sub>t</sub> ± t<sub>α/2, n−2</sub> · s<sub>e</sub> √(1 + 1/n)",
    mathLatex: "\\text{Prediction Interval} = \\hat{Y}_t \\pm t_{\\alpha/2, n-2} \\cdot s_e \\sqrt{1 + \\frac{1}{n}}"
  }
};

function selectPipelineStage(stageKey) {
  const stage = PIPELINE_STAGES[stageKey];
  if (!stage) return;

  // 1. Update pipeline node active states
  document.querySelectorAll('.pipeline-node').forEach(node => {
    const key = node.getAttribute('data-stage');
    if (key === stageKey) {
      node.classList.add('border-cyan-400', 'bg-cyan-500/10', 'glow-cyan');
      node.classList.remove('border-slate-800', 'bg-slate-900/60');
      node.querySelector('.node-dot')?.classList.add('bg-cyan-400', 'shadow-[0_0_12px_#38bdf8]');
      node.querySelector('.node-dot')?.classList.remove('bg-slate-600');
    } else {
      node.classList.remove('border-cyan-400', 'bg-cyan-500/10', 'glow-cyan');
      node.classList.add('border-slate-800', 'bg-slate-900/60');
      node.querySelector('.node-dot')?.classList.remove('bg-cyan-400', 'shadow-[0_0_12px_#38bdf8]');
      node.querySelector('.node-dot')?.classList.add('bg-slate-600');
    }
  });

  // 2. Animate SVG Connecting Line Glow
  const strokePath = document.getElementById('pipeline-flow-path');
  if (strokePath) {
    strokePath.classList.add('stroke-cyan-400');
  }

  // 3. Update Detail Card Content with Smooth Fade-In Transition
  const detailPanel = document.getElementById('pipeline-detail-content');
  if (detailPanel) {
    detailPanel.style.opacity = '0';
    detailPanel.style.transform = 'translateY(8px)';
    detailPanel.style.transition = 'opacity 0.25s ease, transform 0.25s ease';

    let submodelHTML = '';
    if (stage.submodels) {
      submodelHTML = `
        <div class="mt-4 pt-3 border-t border-slate-800/80">
          <p class="text-xs uppercase tracking-wider text-slate-400 font-mono-code mb-2">Explore Individual Architectural Paradigms:</p>
          <div class="grid grid-cols-2 gap-2">
            ${stage.submodels.map(m => `
              <button onclick="previewSubmodel('${m.diagram}', '${m.name}: ${m.desc}')" 
                      class="text-left px-3 py-2 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 transition-all text-xs font-semibold text-slate-200 hover:border-cyan-400">
                <span class="block text-cyan-400">${m.name}</span>
                <span class="text-[11px] text-slate-400 line-clamp-1">${m.desc}</span>
              </button>
            `).join('')}
          </div>
        </div>
      `;
    }

    // Special table CTA button for Phase 5
    let extraCTA = '';
    if (stageKey === 'phase5') {
      extraCTA = `
        <div class="mt-3">
          <a href="#benchmarks-table" 
             class="inline-flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl text-xs font-semibold font-mono-code bg-gradient-to-r from-cyan-500 to-indigo-600 text-white shadow-lg hover:shadow-cyan-500/30 hover:scale-[1.01] transition-all">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/></svg>
            <span>View Full Interactive Benchmark Table Below (12 Studies)</span>
          </a>
        </div>
      `;
    }

    detailPanel.innerHTML = `
      <div class="flex items-center justify-between gap-3 mb-2">
        <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono-code font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
          ${stage.badge}
        </span>
        <button onclick="openDiagramModal('${stage.diagram}', '${stage.diagramCaption}')" 
                class="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium font-mono-code bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 hover:border-cyan-400 transition-all">
          <svg class="w-3.5 h-3.5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"/></svg>
          <span>Inspect 300-DPI Figure</span>
        </button>
      </div>

      <h3 class="text-xl sm:text-2xl font-serif-editorial font-bold text-slate-100">${stage.title}</h3>
      <p class="text-sm font-medium text-cyan-300/90 mt-1">${stage.tagline}</p>
      
      <p class="text-sm text-slate-300 leading-relaxed mt-3">${stage.description}</p>

      <!-- Key Specifications Grid -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mt-4 pt-3 border-t border-slate-800/80">
        ${stage.specs.map(s => `
          <div class="bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/60">
            <div class="text-[11px] font-mono-code text-slate-400">${s.label}</div>
            <div class="text-xs font-semibold text-slate-200 mt-0.5">${s.value}</div>
          </div>
        `).join('')}
      </div>

      <!-- Dedicated Mathematical Formulation Callout Box -->
      <div class="mt-4 p-3.5 rounded-xl bg-slate-900/90 border border-slate-800/90">
        <div class="flex items-center justify-between text-[11px] font-mono-code text-slate-400 mb-1.5">
          <span class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            <span class="font-bold text-slate-200">Mathematical Formulation</span>
          </span>
          <span class="text-[10px] text-cyan-400 font-mono-code">${stage.mathLabel}</span>
        </div>
        <div class="text-xs sm:text-sm font-mono-code text-cyan-300 py-2 px-3 rounded-lg bg-slate-950/80 border border-slate-800 text-center overflow-x-auto shadow-inner">
          ${stage.formula}
        </div>
      </div>

      ${submodelHTML}

      <!-- Enhanced High-Visibility Diagram / Table Preview Container -->
      <div class="mt-4 pt-3 border-t border-slate-800/80">
        <div class="flex items-center justify-between text-xs font-mono-code text-slate-400 mb-1.5">
          <span class="font-semibold text-slate-300">${stage.diagramCaption}</span>
          <span class="text-[11px] text-cyan-400">Click to Open Interactive Zoom</span>
        </div>

        <div class="relative group cursor-pointer overflow-hidden rounded-xl border border-slate-700/80 bg-slate-950 hover:border-cyan-400/80 transition-all shadow-lg"
             onclick="openDiagramModal('${stage.diagram}', '${stage.diagramCaption}')">
          <div class="w-full h-64 sm:h-72 p-3 bg-white flex items-center justify-center overflow-hidden">
            <img id="stage-preview-img" 
                 src="${stage.diagram}" 
                 alt="${stage.diagramCaption}" 
                 class="max-w-full max-h-full object-contain group-hover:scale-[1.03] transition-transform duration-300" />
          </div>
          
          <div class="absolute inset-0 bg-slate-950/70 backdrop-blur-[2px] opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2 text-xs font-semibold text-cyan-300 p-4 text-center">
            <div class="w-10 h-10 rounded-full bg-cyan-500/20 border border-cyan-400 flex items-center justify-center text-cyan-300 shadow-lg">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"/></svg>
            </div>
            <span class="text-slate-100 font-bold">Open Full-Screen High-Resolution Lightbox</span>
            <span class="text-[11px] text-slate-400 font-mono-code font-normal">Supports Pan &amp; Zoom (300 DPI Original Plate)</span>
          </div>
        </div>

        ${extraCTA}
      </div>
    `;

    requestAnimationFrame(() => {
      detailPanel.style.opacity = '1';
      detailPanel.style.transform = 'translateY(0)';
      // Trigger MathJax if available on page
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([detailPanel]).catch(() => {});
      }
    });
  }
}

function previewSubmodel(diagramPath, caption) {
  const img = document.getElementById('stage-preview-img');
  if (img) {
    img.src = diagramPath;
    img.alt = caption;
    img.closest('.group').setAttribute('onclick', `openDiagramModal('${diagramPath}', '${caption.replace(/'/g, "\\'")}')`);
  }
}

// ── ADVANCED LIGHTBOX MODAL ZOOM & PAN CONTROLLER ───────────────────────────
let currentZoom = 1.0;
let panX = 0;
let panY = 0;
let isPanning = false;
let startX = 0;
let startY = 0;

function updateModalTransform() {
  const img = document.getElementById('modal-img');
  const zoomDisplay = document.getElementById('modal-zoom-val');
  if (img) {
    img.style.transform = `translate(${panX}px, ${panY}px) scale(${currentZoom})`;
  }
  if (zoomDisplay) {
    zoomDisplay.textContent = `${Math.round(currentZoom * 100)}%`;
  }
}

function zoomIn() {
  if (currentZoom < 4.0) {
    currentZoom = Math.min(4.0, currentZoom + 0.25);
    updateModalTransform();
  }
}

function zoomOut() {
  if (currentZoom > 0.5) {
    currentZoom = Math.max(0.5, currentZoom - 0.25);
    if (currentZoom <= 1.0) {
      panX = 0;
      panY = 0;
    }
    updateModalTransform();
  }
}

function resetZoom() {
  currentZoom = 1.0;
  panX = 0;
  panY = 0;
  updateModalTransform();
}

function openDiagramModal(imageSrc, captionText) {
  const modal = document.getElementById('diagram-modal');
  const modalImg = document.getElementById('modal-img');
  const modalCaption = document.getElementById('modal-caption');
  const downloadLink = document.getElementById('modal-download');

  if (modal && modalImg) {
    modalImg.src = imageSrc;
    if (modalCaption) modalCaption.textContent = captionText;
    if (downloadLink) downloadLink.href = imageSrc;
    resetZoom();
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }
}

function closeDiagramModal() {
  const modal = document.getElementById('diagram-modal');
  if (modal) {
    modal.classList.add('hidden');
    document.body.style.overflow = 'auto';
    resetZoom();
  }
}

// Global exposure
window.openDiagramModal = openDiagramModal;
window.closeDiagramModal = closeDiagramModal;
window.previewSubmodel = previewSubmodel;
window.selectPipelineStage = selectPipelineStage;

// Event Listeners Setup
document.addEventListener('DOMContentLoaded', () => {
  // Pipeline Node Clicks
  document.querySelectorAll('.pipeline-node').forEach(node => {
    node.addEventListener('click', () => {
      const stage = node.getAttribute('data-stage');
      selectPipelineStage(stage);
    });
  });

  // Modal Controls
  document.getElementById('modal-zoom-in')?.addEventListener('click', zoomIn);
  document.getElementById('modal-zoom-out')?.addEventListener('click', zoomOut);
  document.getElementById('modal-zoom-reset')?.addEventListener('click', resetZoom);
  document.getElementById('modal-close-btn')?.addEventListener('click', closeDiagramModal);

  // Close modal on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeDiagramModal();
      if (window.closePdfModal) window.closePdfModal();
    }
  });

  // Close on backdrop click
  document.getElementById('diagram-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'diagram-modal') {
      closeDiagramModal();
    }
  });

  // Pan & Drag on Modal Image Container
  const container = document.getElementById('modal-viewer-container');
  if (container) {
    container.addEventListener('mousedown', (e) => {
      if (currentZoom > 1.0) {
        isPanning = true;
        startX = e.clientX - panX;
        startY = e.clientY - panY;
        container.classList.add('cursor-grabbing');
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateModalTransform();
    });

    window.addEventListener('mouseup', () => {
      if (isPanning) {
        isPanning = false;
        container.classList.remove('cursor-grabbing');
      }
    });

    // Mouse wheel zoom in modal
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        zoomIn();
      } else {
        zoomOut();
      }
    }, { passive: false });
  }

  // Initial stage selection on page load
  selectPipelineStage('phase1');
});
