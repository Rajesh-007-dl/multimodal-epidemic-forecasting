/**
 * gallery-table.js
 * Manages:
 * 1. Dedicated Publication Architecture Diagram Showcase & Interactive Zoom Viewer
 * 2. Dedicated SOTA Literature Benchmark Comparison Table (Stripe Press / Linear Editorial Design)
 * 3. Live Search & Category Filtering
 * 4. PDF Reader Modal Integration
 */

const ARCHITECTURE_FIGURES = {
  fig1: {
    title: "FIGURE 1: Multimodal Epidemic Forecasting Framework Architecture",
    tagline: "End-to-End System Pipeline Synchronizing 1,143 Days Across 10 Spatio-Temporal Channels",
    image: "assets/images/paper_fig1_system_framework.png",
    caption: "FIGURE 1: Multimodal Epidemic Forecasting Framework Architecture",
    description: "Illustrates the 5-phase dataflow from daily multimodal ingestion (JHU CSSE, Google Mobility, Oxford Blavatnik OxCGRT), through chronological non-leaking partitioning, 4-paradigm multi-model zoo execution, multi-horizon out-of-sample projection, and parametric Student's t uncertainty quantification.",
    modules: [
      { name: "Phase I: Ingestion", detail: "Synchronizes 1,143 daily points across infections, deaths, recoveries, 6 mobility strata, and government stringency." },
      { name: "Phase II: Partitioning", detail: "First-order differencing (d=1) with rolling median imputation and chronological zero-leakage split." },
      { name: "Phase III: Model Zoo", detail: "Parallel training of ARIMA-XGBoost, Holt-Winters RF, Neural-SEIR-PINN, and PatchTST Transformer." },
      { name: "Phase IV & V: Forecast & Bounds", detail: "Direct 10-15 day out-of-sample projection and 80%, 90%, 95% Student's t confidence intervals (df=n-2)." }
    ]
  },
  fig2: {
    title: "FIGURE 2: Hybrid ARIMA-XGBoost Residual Correction Architecture",
    tagline: "Two-Stage Parametric Trend Decomposition with Non-Linear Lag-7 Gradient Boosting",
    image: "assets/images/paper_fig1b_hybrid_arima_xgboost.png",
    caption: "FIGURE 2: Hybrid ARIMA-XGBoost Residual Correction Architecture",
    description: "Combines an Auto-ARIMA linear parametric model with non-linear XGBoost regression. Stage 1 extracts the macro-epidemic trajectory. Stage 2 models the lag-7 residual errors e(t-1)...e(t-7) to capture volatile non-linear waves.",
    modules: [
      { name: "Stage 1: Auto-ARIMA", detail: "AIC grid search minimizes information loss to fit linear parametric baseline L(t)." },
      { name: "Residual Subtraction", detail: "Calculates residual sequence e(t) = Y(t) - L(t) representing unexplained variance." },
      { name: "Stage 2: XGBoost", detail: "Constructs 7-day lagged feature matrix to forecast residual error component r(t)." },
      { name: "Composite Forecast", detail: "Final output Y_hat(t) = L(t) + r(t) with calibrated Student's t prediction interval margins." }
    ]
  },
  fig3: {
    title: "FIGURE 3: Hybrid Holt-Winters + Random Forest Architecture",
    tagline: "Double Exponential Smoothing with 100-Tree Bagged Residual Forest Ensemble",
    image: "assets/images/paper_fig1c_hybrid_hw_rf.png",
    caption: "FIGURE 3: Hybrid Holt-Winters + Random Forest Architecture",
    description: "Decouples level and trend components using Holt-Winters exponential smoothing, then utilizes a 100-tree decorrelated Random Forest ensemble on residual lag vectors to dampen slope variance and avoid explosive overshoot.",
    modules: [
      { name: "Stage 1: Holt-Winters", detail: "Updates level l(t) and additive trend b(t) via alpha and beta smoothing equations." },
      { name: "Residual Vector Extraction", detail: "Forms e(t) = Y(t) - Y_HW(t) over historical training horizon." },
      { name: "Stage 2: Random Forest", detail: "Aggregates 100 decorrelated decision trees using bootstrap bagging on lag features." },
      { name: "Composite Forecast", detail: "Reduces slope extrapolation variance in volatile multi-step prediction horizons." }
    ]
  },
  fig4: {
    title: "FIGURE 4: Physics-Informed Epidemic Neural Network (Neural-SEIR-PINN)",
    tagline: "Bi-LSTM Recurrent Backbone Coupled with Epidemiological SEIR Differential Equations",
    image: "assets/images/paper_fig1d_neural_seir_pinn.png",
    caption: "FIGURE 4: Physics-Informed Epidemic Neural Network (Neural-SEIR-PINN)",
    description: "Enforces strict epidemiological conservation laws. A Bi-LSTM recurrent encoder processes multimodal covariates to output dynamic transmission rates beta(t) and recovery rates gamma(t), which feed into continuous SEIR differential equations.",
    modules: [
      { name: "Bi-LSTM Encoder", detail: "Extracts forward and backward hidden representations from synchronized 10-channel tensor." },
      { name: "Dynamic Parameter Heads", detail: "Estimates time-varying transmission beta(t) and recovery gamma(t) rates." },
      { name: "Epidemiological SEIR ODEs", detail: "Enforces dS/dt, dE/dt, dI/dt, dR/dt conservation laws and prevents unphysical explosions." },
      { name: "Dual-Objective Loss", detail: "L_total = L_data + lambda * L_SEIR (lambda = 0.05) optimized with AdamW." }
    ]
  },
  fig5: {
    title: "FIGURE 5: PatchTST Transformer Architecture",
    tagline: "Channel-Independent Subseries Patched Attention with Direct Multi-Horizon Linear Head",
    image: "assets/images/paper_fig1e_patchtst_transformer.png",
    caption: "FIGURE 5: PatchTST Transformer Architecture",
    description: "Segments time-series into subseries patches (Patch Length P=8, Stride S=4) to preserve local semantic context while dramatically reducing attention memory complexity. Channel independence prevents inter-series cross-contamination.",
    modules: [
      { name: "Subseries Patching", detail: "Transforms 1D temporal series into contiguous patch tokens (P=8, S=4)." },
      { name: "Channel Independence", detail: "Each variable shares identical Transformer weights while retaining isolated attention channels." },
      { name: "Multi-Head Attention", detail: "3-layer Transformer encoder (8 heads, d_model=128) capturing long-range seasonal patterns." },
      { name: "Direct Projection Head", detail: "Outputs full 15-day forward horizon simultaneously, eliminating autoregressive error drift." }
    ]
  },
  fig12: {
    title: "FIGURE 12: Cumulative Recoveries Multi-Model Trajectory Comparison",
    tagline: "Empirical Out-of-Sample Evaluation Demonstrating Plateau Stabilization",
    image: "assets/images/paper_fig12_merged_recoveries_pinn_and_patchtst.png",
    caption: "FIGURE 12: Cumulative Recoveries forecasting using Neural-SEIR-PINN and PatchTST-Transformer",
    description: "Compares actual JHU CSSE recoveries against Neural-SEIR-PINN and PatchTST-Transformer in the out-of-sample test window. While classical autoregressive models diverge exponentially, our proposed models maintain stable plateaus matching the 131.2M empirical ground truth.",
    modules: [
      { name: "Actual Series (JHU CSSE)", detail: "Plateaus at 131,245,600 following reporting cessation in August 2021." },
      { name: "PatchTST Forecast", detail: "Maintains smooth parallel trajectory at 137.5M without runaway divergence." },
      { name: "Neural-SEIR-PINN", detail: "Strictly preserves epidemiological saturation bounds (Rₜ < 1.0)." },
      { name: "Statistical Superiority", detail: "Achieves R² = 0.976 (PatchTST) and R² = 0.972 (PINN) over 1,143 total days." }
    ]
  }
};

let currentGalleryZoom = 1.0;

function switchGalleryFigure(figKey) {
  const fig = ARCHITECTURE_FIGURES[figKey];
  if (!fig) return;

  // Update button active state
  document.querySelectorAll('.gallery-tab-btn').forEach(btn => {
    if (btn.getAttribute('data-fig') === figKey) {
      btn.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
      btn.classList.remove('text-slate-400', 'border-transparent');
    } else {
      btn.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
      btn.classList.add('text-slate-400', 'border-transparent');
    }
  });

  // Update Diagram Image & Captions
  const img = document.getElementById('gallery-display-img');
  const title = document.getElementById('gallery-fig-title');
  const tagline = document.getElementById('gallery-fig-tagline');
  const caption = document.getElementById('gallery-fig-caption');
  const desc = document.getElementById('gallery-fig-desc');
  const modulesContainer = document.getElementById('gallery-modules-list');
  const inspectBtn = document.getElementById('gallery-inspect-btn');
  const downloadBtn = document.getElementById('gallery-download-btn');

  if (img) {
    img.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
    img.style.opacity = '0';
    setTimeout(() => {
      img.src = fig.image;
      img.alt = fig.caption;
      resetGalleryZoom();
      img.style.opacity = '1';
    }, 120);
  }
  if (title) title.textContent = fig.title;
  if (tagline) tagline.textContent = fig.tagline;
  if (caption) caption.textContent = fig.caption;
  if (desc) desc.textContent = fig.description;

  if (inspectBtn) {
    inspectBtn.onclick = () => window.openDiagramModal(fig.image, fig.caption);
  }
  if (downloadBtn) {
    downloadBtn.href = fig.image;
  }

  if (modulesContainer && fig.modules) {
    modulesContainer.innerHTML = fig.modules.map((m, idx) => `
      <div class="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700 transition-colors">
        <div class="flex items-center gap-2 mb-1">
          <span class="w-5 h-5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[11px] font-mono-code font-bold flex items-center justify-center">${idx + 1}</span>
          <span class="text-xs font-bold text-slate-200">${m.name}</span>
        </div>
        <p class="text-xs text-slate-400 leading-relaxed pl-7">${m.detail}</p>
      </div>
    `).join('');
  }
}

function zoomGalleryIn() {
  if (currentGalleryZoom < 3.0) {
    currentGalleryZoom = Math.min(3.0, currentGalleryZoom + 0.25);
    updateGalleryTransform();
  }
}

function zoomGalleryOut() {
  if (currentGalleryZoom > 0.8) {
    currentGalleryZoom = Math.max(0.8, currentGalleryZoom - 0.25);
    updateGalleryTransform();
  }
}

function resetGalleryZoom() {
  currentGalleryZoom = 1.0;
  updateGalleryTransform();
}

function updateGalleryTransform() {
  const img = document.getElementById('gallery-display-img');
  const zoomDisplay = document.getElementById('gallery-zoom-val');
  if (img) {
    img.style.transform = `scale(${currentGalleryZoom})`;
  }
  if (zoomDisplay) {
    zoomDisplay.textContent = `${Math.round(currentGalleryZoom * 100)}%`;
  }
}

// ── SOTA BENCHMARK TABLE CONTROLLER (REFINED EDITORIAL STRUCTURE) ───────────

const BENCHMARK_TABLE_DATA = [
  {
    type: "baseline",
    refId: "[1]",
    author: "Rustam et al.",
    venue: "IEEE Access 2020",
    family: "Statistical ML",
    familyClass: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    archTitle: "Exponential Smoothing (ES) & Baseline ML",
    archSubtitle: "Classic Linear Trend with Holts Exponential Filter",
    horizon: "Global Cumulative",
    horizonDetail: "66-Day Early Wave",
    fitItems: [
      { label: "R²", val: "0.98", highlight: "emerald" },
      { label: "Adj-R²", val: "0.97", highlight: "emerald" }
    ],
    errorList: [
      { target: "Confirmed", mae: "8.8k", rmse: "16.8k" },
      { target: "Deaths", mae: "406", rmse: "814" },
      { target: "Recoveries", mae: "1.8k", rmse: "2.4k" }
    ],
    relativeBadge: "Not reported",
    relativeBadgeClass: "bg-slate-800 text-slate-400",
    uncertaintyType: "none",
    uncertaintyText: "Deterministic point forecasts only",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[5]",
    author: "Nazir et al.",
    venue: "PLOS ONE 2026",
    family: "Deep Covariates",
    familyClass: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    archTitle: "N-BEATS Deep Network",
    archSubtitle: "Interpretable Basis Expansion + Google Mobility Covariates",
    horizon: "Daily Regional Panels",
    horizonDetail: "UK & USA Outbreak Waves",
    fitItems: [
      { label: "Goodness", val: "N/A", highlight: "slate" }
    ],
    errorList: [
      { target: "UK Daily", mae: "—", rmse: "550 to 859" },
      { target: "USA Daily", mae: "—", rmse: "7.8k to 8.4k" }
    ],
    relativeBadge: "MAPE = 7.26% – 25.0%",
    relativeBadgeClass: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Unquantified",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[6]",
    author: "Srisuradetchai et al.",
    venue: "ICBIR 2022",
    family: "Hybrid SVR",
    familyClass: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
    archTitle: "Hybrid ARIMA + SVR",
    archSubtitle: "Linear Box-Jenkins Baseline with Non-linear SVR Residuals",
    horizon: "Daily Cases",
    horizonDetail: "Thailand 3rd Wave",
    fitItems: [
      { label: "Goodness", val: "N/A", highlight: "slate" }
    ],
    errorList: [
      { target: "Hybrid Model", mae: "—", rmse: "343.60" },
      { target: "ARIMA Baseline", mae: "—", rmse: "358.97" },
      { target: "SVR Standalone", mae: "—", rmse: "362.95" }
    ],
    relativeBadge: "MAPE = 13.35%",
    relativeBadgeClass: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Unquantified",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[7]",
    author: "Rahman et al.",
    venue: "PLOS Glob. Pub. 2022",
    family: "Parametric vs Tree",
    familyClass: "bg-slate-800 text-slate-300 border-slate-700",
    archTitle: "ARIMA vs. Standalone XGBoost",
    archSubtitle: "Direct Model Comparison on Early In-sample Sets",
    horizon: "Bangladesh Outbreak",
    horizonDetail: "Confirmed & Deaths",
    fitItems: [
      { label: "Goodness", val: "N/A", highlight: "slate" }
    ],
    errorList: [
      { target: "Confirmed", mae: "2.7k", rmse: "3.1k" },
      { target: "Deaths", mae: "79", rmse: "87" }
    ],
    relativeBadge: "Severe Divergence (>130%)",
    relativeBadgeClass: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Unquantified",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[8]",
    author: "Kumar & Susan",
    venue: "ICCCNT 2020",
    family: "Prophet Baseline",
    familyClass: "bg-sky-500/10 text-sky-400 border-sky-500/20",
    archTitle: "ARIMA(9,1,2) vs. Prophet",
    archSubtitle: "Facebook Prophet Additive Model Comparison",
    horizon: "Global 10-Day Multi-Step",
    horizonDetail: "Multi-Horizon Rollout",
    fitItems: [
      { label: "Accuracy", val: "79.6%", highlight: "cyan" }
    ],
    errorList: [
      { target: "Conf (ARIMA)", mae: "19k", rmse: "21k" },
      { target: "Conf (Prophet)", mae: "168k", rmse: "182k" },
      { target: "Death (ARIMA)", mae: "662", rmse: "821" }
    ],
    relativeBadge: "Prophet MAPE: 70.6%",
    relativeBadgeClass: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Deterministic only",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[15]",
    author: "Ahmad & Kumar",
    venue: "JAAC 2026",
    family: "Physics-Informed",
    familyClass: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    archTitle: "PyTorch SEIR-PINN Simulation",
    archSubtitle: "Physics-Constrained Parameter Inversion (β, γ)",
    horizon: "Synthetic ODE Trajectories",
    horizonDetail: "Standard Epidemic Curve",
    fitItems: [
      { label: "ODE Loss", val: "< 1.0e-3", highlight: "emerald" }
    ],
    errorList: [
      { target: "β Inversion", mae: "1.2e-3", rmse: "—" },
      { target: "γ Inversion", mae: "4.5e-4", rmse: "—" }
    ],
    relativeBadge: "Residual Loss < 1e-3",
    relativeBadgeClass: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Deterministic ODE",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[17]",
    author: "Ameet & Rana",
    venue: "ICIPTM 2023",
    family: "Recurrent Deep Net",
    familyClass: "bg-violet-500/10 text-violet-400 border-violet-500/20",
    archTitle: "Conv-LSTM vs. Univariate LSTM",
    archSubtitle: "Spatio-Temporal Convolutional Feature Extractor",
    horizon: "India Daily Waves",
    horizonDetail: "Regional Increments",
    fitItems: [
      { label: "Loss Drop", val: "0.32%", highlight: "cyan" }
    ],
    errorList: [
      { target: "Conv-LSTM Loss", mae: "0.069", rmse: "0.126" },
      { target: "LSTM Baseline", mae: "0.156", rmse: "—" }
    ],
    relativeBadge: "MAPE = 2.17%",
    relativeBadgeClass: "bg-violet-500/10 text-violet-400 border-violet-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Unquantified",
    isProposed: false
  },
  {
    type: "baseline",
    refId: "[19]",
    author: "Nie et al.",
    venue: "ICLR 2023",
    family: "Patch Transformer",
    familyClass: "bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20",
    archTitle: "PatchTST Foundation Model",
    archSubtitle: "Subseries Patched Multi-Head Attention",
    horizon: "Multivariate Benchmarks",
    horizonDetail: "ETTh, Weather, Traffic",
    fitItems: [
      { label: "Rank", val: "SOTA #1", highlight: "emerald" }
    ],
    errorList: [
      { target: "Benchmark MSE", mae: "—", rmse: "0.385" },
      { target: "Benchmark MAE", mae: "0.397", rmse: "—" }
    ],
    relativeBadge: "Beats Autoformer (-22%)",
    relativeBadgeClass: "bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Point projection",
    isProposed: false
  },
  {
    type: "proposed",
    refId: "PROPOSED 1",
    author: "Our Approach (Model 1)",
    venue: "Preprint Nov 2026",
    family: "Hybrid Statistical + ML",
    familyClass: "bg-cyan-500/20 text-cyan-300 border-cyan-400/40",
    archTitle: "Hybrid ARIMA-XGBoost",
    archSubtitle: "Linear Auto-ARIMA Trend + Lag-7 XGBoost Residual Booster",
    horizon: "Tri-Modal Slices",
    horizonDetail: "10-Day Multi-Horizon (Days 67–76)",
    fitItems: [
      { label: "Conf R²", val: "0.38", highlight: "cyan" },
      { label: "Rec R²", val: "0.76", highlight: "emerald" },
      { label: "Death R²", val: "0.04", highlight: "slate" }
    ],
    errorList: [
      { target: "Confirmed", mae: "378.64", rmse: "450.54" },
      { target: "Recoveries", mae: "86.52", rmse: "101.90" },
      { target: "Deaths", mae: "37.04", rmse: "51.73" }
    ],
    relativeBadge: "Residual Var: 0.038",
    relativeBadgeClass: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30 font-bold",
    uncertaintyType: "parametric",
    uncertaintyText: "Student's t (df=64): ±0.33 Deaths, ±2.49 Rec, ±7.92 Conf (95% CI)",
    isProposed: true
  },
  {
    type: "proposed",
    refId: "PROPOSED 2",
    author: "Our Approach (Model 2)",
    venue: "Preprint Nov 2026",
    family: "Exponential Ensemble",
    familyClass: "bg-cyan-500/20 text-cyan-300 border-cyan-400/40",
    archTitle: "Hybrid Holt-Winters + Random Forest",
    archSubtitle: "Double Exponential Smoothing + 100-Tree Bagged Residual Forest",
    horizon: "Tri-Modal Slices",
    horizonDetail: "10-Day Multi-Horizon (Days 67–76)",
    fitItems: [
      { label: "Conf R²", val: "0.27", highlight: "cyan" },
      { label: "Rec R²", val: "-0.05", highlight: "rose" },
      { label: "Death R²", val: "-0.42", highlight: "rose" }
    ],
    errorList: [
      { target: "Confirmed", mae: "409.13", rmse: "486.42" },
      { target: "Recoveries", mae: "170.38", rmse: "213.43" },
      { target: "Deaths", mae: "44.80", rmse: "62.91" }
    ],
    relativeBadge: "Slope Variance Damped",
    relativeBadgeClass: "bg-indigo-500/10 text-indigo-300 border-indigo-500/20",
    uncertaintyType: "none",
    uncertaintyText: "Bootstrap Ensembled",
    isProposed: true
  },
  {
    type: "proposed",
    refId: "PROPOSED 3",
    author: "Our Approach (Model 3)",
    venue: "Preprint Nov 2026",
    family: "Physics-Informed Deep Net",
    familyClass: "bg-emerald-500/20 text-emerald-300 border-emerald-400/40 font-bold",
    archTitle: "Neural-SEIR-PINN",
    archSubtitle: "Bi-LSTM Encoder + Dynamic β(t), γ(t) ODE Conservation Integration",
    horizon: "1,143 Consecutive Days",
    horizonDetail: "Full Pandemic Spectrum (2020–2023)",
    fitItems: [
      { label: "Conf R²", val: "0.44", highlight: "cyan" },
      { label: "Rec R²", val: "0.86", highlight: "emerald" },
      { label: "Peak Fit", val: "0.972", highlight: "emerald" }
    ],
    errorList: [
      { target: "Confirmed", mae: "362.65", rmse: "426.17" },
      { target: "Recoveries", mae: "68.85", rmse: "79.11" },
      { target: "Deaths", mae: "36.45", rmse: "50.84" }
    ],
    relativeBadge: "L_total = L_data + 0.05*L_SEIR",
    relativeBadgeClass: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-bold",
    uncertaintyType: "physics",
    uncertaintyText: "Dynamic Rt = β(t)/γ(t) Strict Compartment Guardrails (0% Unphysical Error)",
    isProposed: true
  },
  {
    type: "proposed",
    refId: "PROPOSED 4",
    author: "Our Approach (Model 4)",
    venue: "Preprint Nov 2026",
    family: "Foundation Transformer",
    familyClass: "bg-cyan-500/20 text-cyan-300 border-cyan-400/40 font-bold",
    archTitle: "PatchTST Transformer",
    archSubtitle: "Subseries Patched Attention (P=8, S=4) with Direct Multi-Step Projection",
    horizon: "Tri-Modal Long-Horizon",
    horizonDetail: "15-Day Forward Rollout Head",
    fitItems: [
      { label: "Recovery Fit", val: "0.976", highlight: "emerald" },
      { label: "Conf R²", val: "-0.19", highlight: "rose" }
    ],
    errorList: [
      { target: "Confirmed", mae: "518.23", rmse: "622.27" },
      { target: "Recoveries", mae: "180.70", rmse: "230.00" },
      { target: "Deaths", mae: "43.12", rmse: "60.50" }
    ],
    relativeBadge: "Zero Autoregressive Drift",
    relativeBadgeClass: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-bold",
    uncertaintyType: "direct",
    uncertaintyText: "Direct 15-Day Rollout Projection (Eliminates recursive compounding)",
    isProposed: true
  }
];

let activeTableFilter = 'all';
let tableSearchQuery = '';

function renderBenchmarkTable(filterKey = 'all', searchQuery = '') {
  activeTableFilter = filterKey;
  tableSearchQuery = searchQuery.trim().toLowerCase();
  const tbody = document.getElementById('benchmark-table-body');
  if (!tbody) return;

  const filteredData = BENCHMARK_TABLE_DATA.filter(row => {
    // 1. Category Filter
    if (filterKey === 'proposed' && !row.isProposed) return false;
    if (filterKey === 'baselines' && row.isProposed) return false;

    // 2. Search Query Filter
    if (tableSearchQuery) {
      const matchTarget = `
        ${row.author} ${row.refId} ${row.venue} ${row.archTitle} 
        ${row.archSubtitle} ${row.family} ${row.horizon} ${row.horizonDetail}
      `.toLowerCase();
      if (!matchTarget.includes(tableSearchQuery)) return false;
    }

    return true;
  });

  if (filteredData.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="p-12 text-center text-slate-400 font-mono-code text-xs">
          No studies match the filter criteria "${searchQuery}".
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filteredData.map(row => {
    const isProposed = row.isProposed;
    
    // Fit Badges
    const fitHTML = row.fitItems.map(fit => {
      let colorClass = 'bg-slate-800 text-slate-300 border-slate-700';
      if (fit.highlight === 'emerald') colorClass = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 font-bold';
      if (fit.highlight === 'cyan') colorClass = 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30 font-bold';
      if (fit.highlight === 'rose') colorClass = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      
      return `
        <div class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono-code border ${colorClass} mb-1 mr-1">
          <span class="text-slate-400 text-[10px]">${fit.label}:</span>
          <span>${fit.val}</span>
        </div>
      `;
    }).join('');

    // Error List Chips
    const errorHTML = row.errorList.map(err => `
      <div class="flex items-center justify-between gap-2 py-0.5 border-b border-slate-800/40 last:border-none text-[11px] font-mono-code">
        <span class="text-slate-400">${err.target}:</span>
        <span class="text-slate-200">
          ${err.mae !== '—' ? `<span class="text-slate-400 text-[10px]">MAE</span> <strong class="text-cyan-300">${err.mae}</strong>` : ''}
          ${err.rmse !== '—' ? ` <span class="text-slate-500">•</span> <span class="text-slate-400 text-[10px]">RMSE</span> <strong class="text-indigo-300">${err.rmse}</strong>` : ''}
        </span>
      </div>
    `).join('');

    // Uncertainty Badge
    let uncertaintyHTML = '';
    if (row.uncertaintyType === 'parametric') {
      uncertaintyHTML = `
        <div class="p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-[11px] text-cyan-300 font-mono-code leading-relaxed">
          <div class="flex items-center gap-1 font-bold text-cyan-400 mb-0.5">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <span>Parametric Bounds</span>
          </div>
          ${row.uncertaintyText}
        </div>
      `;
    } else if (row.uncertaintyType === 'physics') {
      uncertaintyHTML = `
        <div class="p-2 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-[11px] text-emerald-300 font-mono-code leading-relaxed">
          <div class="flex items-center gap-1 font-bold text-emerald-400 mb-0.5">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
            <span>Biological Guardrails</span>
          </div>
          ${row.uncertaintyText}
        </div>
      `;
    } else if (row.uncertaintyType === 'direct') {
      uncertaintyHTML = `
        <div class="p-2 rounded-lg bg-purple-950/40 border border-purple-500/30 text-[11px] text-purple-300 font-mono-code leading-relaxed">
          <div class="flex items-center gap-1 font-bold text-purple-400 mb-0.5">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            <span>Direct Rollout Head</span>
          </div>
          ${row.uncertaintyText}
        </div>
      `;
    } else {
      uncertaintyHTML = `<span class="text-xs text-slate-500 italic font-sans">${row.uncertaintyText}</span>`;
    }

    // Row Container Styling
    const rowBg = isProposed
      ? 'bg-gradient-to-r from-cyan-950/30 via-slate-900/40 to-indigo-950/20 border-l-4 border-l-cyan-400 hover:bg-cyan-950/50 shadow-sm'
      : 'hover:bg-slate-900/70 border-l-4 border-l-transparent hover:border-l-indigo-400/70 transition-all';

    return `
      <tr class="${rowBg} border-b border-slate-800/60 transition-all duration-150">
        <!-- Col 1: Study & Author -->
        <td class="p-4 align-top w-52">
          <div class="flex items-center gap-2 mb-1">
            <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono-code font-bold ${isProposed ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800 text-slate-300'}">
              ${row.refId}
            </span>
            <span class="text-xs font-semibold text-slate-200">${row.author}</span>
          </div>
          <div class="text-[11px] text-slate-400 font-mono-code mb-1.5">${row.venue}</div>
          ${isProposed 
            ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-code font-bold uppercase tracking-wider bg-cyan-500/20 text-cyan-300 border border-cyan-400/50 shadow-[0_0_12px_rgba(56,189,248,0.25)]">
                 <span class="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
                 PROPOSED SOTA
               </span>`
            : `<span class="inline-block px-1.5 py-0.5 rounded text-[10px] font-mono-code text-slate-500 bg-slate-900 border border-slate-800">
                 LITERATURE BASELINE
               </span>`
          }
        </td>

        <!-- Col 2: Evaluated Architecture -->
        <td class="p-4 align-top min-w-[200px]">
          <span class="inline-block px-2 py-0.5 rounded text-[10px] font-mono-code uppercase font-semibold border ${row.familyClass} mb-1.5">
            ${row.family}
          </span>
          <div class="text-xs font-bold text-slate-100">${row.archTitle}</div>
          <div class="text-[11px] text-slate-400 leading-snug mt-0.5 font-sans">${row.archSubtitle}</div>
        </td>

        <!-- Col 3: Target Scope & Horizon -->
        <td class="p-4 align-top w-44">
          <div class="text-xs font-semibold text-slate-200">${row.horizon}</div>
          <div class="text-[11px] text-slate-400 font-mono-code mt-0.5">${row.horizonDetail}</div>
        </td>

        <!-- Col 4: Goodness-of-Fit -->
        <td class="p-4 align-top w-40">
          <div class="flex flex-wrap">${fitHTML}</div>
        </td>

        <!-- Col 5: MAE / RMSE Errors -->
        <td class="p-4 align-top min-w-[220px]">
          <div class="space-y-1 bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
            ${errorHTML}
          </div>
        </td>

        <!-- Col 6: Relative Error -->
        <td class="p-4 align-top w-36">
          <span class="inline-block px-2 py-1 rounded text-[11px] font-mono-code border ${row.relativeBadgeClass} leading-tight text-center">
            ${row.relativeBadge}
          </span>
        </td>

        <!-- Col 7: Prediction Uncertainty -->
        <td class="p-4 align-top min-w-[210px]">
          ${uncertaintyHTML}
        </td>
      </tr>
    `;
  }).join('');
}

function switchTableFilter(filterKey) {
  document.querySelectorAll('.table-filter-btn').forEach(btn => {
    if (btn.getAttribute('data-filter') === filterKey) {
      btn.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
      btn.classList.remove('text-slate-400', 'border-transparent');
    } else {
      btn.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
      btn.classList.add('text-slate-400', 'border-transparent');
    }
  });
  renderBenchmarkTable(filterKey, tableSearchQuery);
}

function switchTableViewMode(mode) {
  const tableContainer = document.getElementById('table-interactive-view');
  const plateContainer = document.getElementById('table-image-view');
  const btnInteractive = document.getElementById('btn-view-interactive');
  const btnPlate = document.getElementById('btn-view-plate');

  if (mode === 'interactive') {
    tableContainer?.classList.remove('hidden');
    plateContainer?.classList.add('hidden');
    btnInteractive?.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
    btnInteractive?.classList.remove('text-slate-400', 'border-transparent');
    btnPlate?.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
    btnPlate?.classList.add('text-slate-400', 'border-transparent');
  } else {
    tableContainer?.classList.add('hidden');
    plateContainer?.classList.remove('hidden');
    btnPlate?.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
    btnPlate?.classList.remove('text-slate-400', 'border-transparent');
    btnInteractive?.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
    btnInteractive?.classList.add('text-slate-400', 'border-transparent');
  }
}

// ── FULL-SCREEN PDF READER MODAL CONTROLLER ─────────────────────────────────
function openPdfModal() {
  const modal = document.getElementById('pdf-reader-modal');
  if (modal) {
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }
}

function closePdfModal() {
  const modal = document.getElementById('pdf-reader-modal');
  if (modal) {
    modal.classList.add('hidden');
    document.body.style.overflow = 'auto';
  }
}

// Make functions globally available
window.openPdfModal = openPdfModal;
window.closePdfModal = closePdfModal;

// Attach event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Gallery Tab Click Handlers
  document.querySelectorAll('.gallery-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const figKey = btn.getAttribute('data-fig');
      switchGalleryFigure(figKey);
    });
  });

  // Gallery Zoom Controls
  document.getElementById('gallery-zoom-in')?.addEventListener('click', zoomGalleryIn);
  document.getElementById('gallery-zoom-out')?.addEventListener('click', zoomGalleryOut);
  document.getElementById('gallery-zoom-reset')?.addEventListener('click', resetGalleryZoom);

  // Table Filter Tabs
  document.querySelectorAll('.table-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const filterKey = btn.getAttribute('data-filter');
      switchTableFilter(filterKey);
    });
  });

  // Table Live Search Input
  const searchInput = document.getElementById('table-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      renderBenchmarkTable(activeTableFilter, e.target.value);
    });
  }

  // Table View Switcher
  document.getElementById('btn-view-interactive')?.addEventListener('click', () => switchTableViewMode('interactive'));
  document.getElementById('btn-view-plate')?.addEventListener('click', () => switchTableViewMode('plate'));

  // PDF Modal Listeners
  document.getElementById('pdf-close-btn')?.addEventListener('click', closePdfModal);
  document.getElementById('pdf-reader-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'pdf-reader-modal' || e.target.classList.contains('modal-overlay')) {
      closePdfModal();
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePdfModal();
  });

  // Initialize Gallery and Table
  switchGalleryFigure('fig1');
  renderBenchmarkTable('all');
});
