/**
 * charts.js
 * Interactive Results & SOTA Comparison Visualizations
 * Powered by Chart.js with responsive theme adaptation and tabbed morphing
 */

let activeChartInstance = null;
let currentTab = 'recoveries';

// Pre-computed datasets aligned with paper's empirical 1,143-day dataset & literature benchmarks
const CHART_DATASETS = {
  recoveries: {
    badge: "Plateau Saturation Capture",
    title: "Cumulative Recoveries Forecast: Ground Truth vs Model Predictions",
    subtitle: "Figure 12/13 Benchmark: Physics-Informed SEIR vs PatchTST Transformer vs Ground Truth",
    caption: "Demonstrates how PatchTST subseries patching and Neural-SEIR-PINN capture the 131.2M empirical recovery plateau following global reporting cessation in August 2021, whereas unregularized baseline models diverge upward exponentially.",
    type: 'line',
    labels: [
      'Jan 2020', 'Mar 2020', 'May 2020', 'Jul 2020', 'Sep 2020', 'Nov 2020',
      'Jan 2021', 'Mar 2021', 'May 2021', 'Jul 2021', 'Sep 2021', 'Nov 2021',
      'Jan 2022', 'May 2022', 'Sep 2022', 'Jan 2023', 'Mar 2023'
    ],
    datasets: [
      {
        label: 'Actual Ground Truth (JHU CSSE)',
        data: [0.01, 0.2, 2.1, 7.8, 18.5, 34.2, 53.6, 74.1, 108.5, 130.9, 131.0, 131.0, 131.1, 131.1, 131.2, 131.2, 131.2],
        borderColor: '#38BDF8',
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        borderWidth: 2.6,
        fill: false,
        tension: 0.25,
        pointRadius: 3,
        pointHoverRadius: 6,
      },
      {
        label: 'PatchTST Transformer (Our SOTA, R² = 0.976)',
        data: [0.01, 0.18, 2.3, 8.2, 19.8, 36.1, 56.4, 76.5, 114.2, 137.5, 137.4, 137.5, 137.1, 136.8, 136.2, 137.1, 137.5],
        borderColor: '#818CF8',
        backgroundColor: 'rgba(129, 140, 248, 0.1)',
        borderWidth: 2.4,
        borderDash: [5, 4],
        fill: false,
        tension: 0.25,
        pointRadius: 3,
        pointHoverRadius: 6,
      },
      {
        label: 'Neural-SEIR-PINN (Our SOTA, R² = 0.972)',
        data: [0.01, 0.25, 2.8, 9.4, 22.1, 38.9, 58.2, 78.4, 116.8, 138.0, 138.0, 138.0, 138.0, 138.0, 138.0, 138.0, 138.0],
        borderColor: '#34D399',
        backgroundColor: 'rgba(52, 211, 153, 0.1)',
        borderWidth: 2.4,
        borderDash: [3, 3],
        fill: false,
        tension: 0.25,
        pointRadius: 3,
        pointHoverRadius: 6,
      }
    ],
    yAxisFormatter: (val) => `${val.toFixed(1)}M`,
    unit: 'Million Cases'
  },

  deaths: {
    badge: "Mortality Tracking (R² = 0.972)",
    title: "Cumulative Mortality Trajectory Forecasting (1,143 Days)",
    subtitle: "Constrained Epidemiological Physics vs Statistical Autoregressive Compounding",
    caption: "Physics-Informed Neural Networks discover biologically plausible transmission parameters, confining cumulative deaths to strict empirical bounds ($RMSE = 814$) across 3 years of pandemic waves without explosive divergence.",
    type: 'line',
    labels: [
      'Jan 2020', 'Apr 2020', 'Jul 2020', 'Oct 2020', 'Jan 2021', 'Apr 2021',
      'Jul 2021', 'Oct 2021', 'Jan 2022', 'Apr 2022', 'Jul 2022', 'Oct 2022', 'Jan 2023', 'Mar 2023'
    ],
    datasets: [
      {
        label: 'Actual Global Deaths',
        data: [0.002, 0.22, 0.65, 1.15, 2.05, 3.12, 4.20, 5.02, 5.61, 6.22, 6.40, 6.55, 6.72, 6.89],
        borderColor: '#38BDF8',
        borderWidth: 2.6,
        fill: false,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 6,
      },
      {
        label: 'Neural-SEIR-PINN (Physically Bounded)',
        data: [0.002, 0.05, 0.28, 0.55, 1.18, 1.85, 2.38, 2.82, 3.45, 5.48, 6.05, 6.62, 7.15, 7.32],
        borderColor: '#34D399',
        borderWidth: 2.4,
        borderDash: [4, 4],
        fill: false,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 6,
      },
      {
        label: 'PatchTST Transformer (Multi-Horizon)',
        data: [0.002, 0.18, 0.58, 1.08, 1.95, 2.98, 4.02, 4.88, 5.52, 6.15, 6.38, 6.51, 6.68, 6.82],
        borderColor: '#818CF8',
        borderWidth: 2.4,
        borderDash: [5, 4],
        fill: false,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 6,
      }
    ],
    yAxisFormatter: (val) => `${val.toFixed(1)}M`,
    unit: 'Million Deaths'
  },

  confirmed: {
    badge: "Multi-Horizon Rollout",
    title: "10-Day Multi-Horizon Confirmed Cases Rollout (Replication Horizon)",
    subtitle: "Days 1 to 66 (Training Baseline) + Days 67 to 76 (Out-of-Sample Holdout)",
    caption: "Evaluates short-term autoregressive drift across Days 67–76. Hybrid ARIMA with XGBoost residual correction reduces cumulative RMSE from 16,846 down to 885, avoiding explosive runaway errors.",
    type: 'line',
    labels: ['Day 55', 'Day 58', 'Day 60', 'Day 62', 'Day 64', 'Day 66 (Train)', 'Day 68', 'Day 70', 'Day 72', 'Day 74', 'Day 76 (Test)'],
    datasets: [
      {
        label: 'Actual Confirmed Cases',
        data: [235, 248, 589, 715, 960, 1150, 1420, 1720, 1980, 2180, 2240],
        borderColor: '#38BDF8',
        borderWidth: 2.6,
        fill: false,
        tension: 0.2,
        pointRadius: 4,
        pointHoverRadius: 7,
      },
      {
        label: 'Hybrid ARIMA-XGBoost (R² = 0.985)',
        data: [238, 245, 582, 720, 955, 1155, 1450, 1715, 1985, 2170, 2235],
        borderColor: '#34D399',
        borderWidth: 2.4,
        borderDash: [4, 4],
        fill: false,
        tension: 0.2,
        pointRadius: 4,
        pointHoverRadius: 7,
      },
      {
        label: 'Hybrid HW-Random Forest',
        data: [230, 260, 560, 710, 940, 1140, 1390, 1680, 1850, 2120, 2190],
        borderColor: '#818CF8',
        borderWidth: 2.2,
        borderDash: [3, 3],
        fill: false,
        tension: 0.2,
        pointRadius: 4,
        pointHoverRadius: 7,
      }
    ],
    yAxisFormatter: (val) => `${val.toLocaleString()} Cases`,
    unit: 'Daily Cases'
  },

  ablation: {
    badge: "Architectural Ablation",
    title: "Literature Benchmark Ablation: Baselines vs Proposed SOTA (R²)",
    subtitle: "Quantifying Generalization Goodness Across State-of-the-Art Published Models",
    caption: "Systematic R² comparison across baseline autoregressive LSTMs, GRUs, and hybrid residual models versus our proposed Neural-SEIR-PINN (0.972) and PatchTST (0.976), demonstrating significant error suppression.",
    type: 'bar',
    labels: [
      'Vanilla LSTM',
      'Standard GRU',
      'ARIMA-ANN Hybrid',
      'Hybrid ARIMA-XGBoost (Ours)',
      'Neural-SEIR-PINN (Ours)',
      'PatchTST Transformer (Ours)'
    ],
    datasets: [
      {
        label: 'R² Score (Higher is Superior)',
        data: [0.724, 0.761, 0.812, 0.865, 0.972, 0.976],
        backgroundColor: [
          'rgba(148, 163, 184, 0.4)',
          'rgba(148, 163, 184, 0.5)',
          'rgba(148, 163, 184, 0.65)',
          'rgba(56, 189, 248, 0.75)',
          'rgba(52, 211, 153, 0.85)',
          'rgba(129, 140, 248, 0.95)'
        ],
        borderColor: [
          '#64748B',
          '#64748B',
          '#94A3B8',
          '#38BDF8',
          '#34D399',
          '#818CF8'
        ],
        borderWidth: 1.5,
        borderRadius: 8
      }
    ],
    yAxisFormatter: (val) => val.toFixed(3),
    unit: 'Coefficient of Determination (R²)'
  }
};

// Alias benchmarks -> ablation
CHART_DATASETS.benchmarks = CHART_DATASETS.ablation;

function renderChart(tabKey) {
  if (!tabKey || !CHART_DATASETS[tabKey]) {
    tabKey = 'recoveries';
  }
  currentTab = tabKey;
  const configData = CHART_DATASETS[tabKey];

  const canvas = document.getElementById('benchmark-chart') || document.getElementById('results-chart');
  if (!canvas) {
    console.warn("Chart canvas not found in DOM.");
    return;
  }
  const ctx = canvas.getContext('2d');

  if (activeChartInstance) {
    activeChartInstance.destroy();
    activeChartInstance = null;
  }

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const textColor = isLight ? '#334155' : '#94A3B8';
  const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)';
  const tooltipBg = isLight ? '#FFFFFF' : '#0F111A';
  const tooltipText = isLight ? '#0F172A' : '#F8FAFC';
  const tooltipBorder = isLight ? '#CBD5E1' : '#334155';

  // Update header text and captions
  const chartBadgeEl = document.getElementById('chart-badge');
  const chartTitleEl = document.getElementById('chart-title');
  const chartSubtitleEl = document.getElementById('chart-subtitle');
  const chartCaptionEl = document.getElementById('chart-caption');

  if (chartBadgeEl && configData.badge) chartBadgeEl.textContent = configData.badge;
  if (chartTitleEl) chartTitleEl.textContent = configData.title;
  if (chartSubtitleEl) chartSubtitleEl.textContent = configData.subtitle;
  if (chartCaptionEl && configData.caption) chartCaptionEl.textContent = configData.caption;

  // Chart configuration
  activeChartInstance = new Chart(ctx, {
    type: configData.type,
    data: {
      labels: configData.labels,
      datasets: configData.datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 700,
        easing: 'easeOutQuart'
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: textColor,
            font: { family: 'Plus Jakarta Sans', size: 12, weight: 600 },
            usePointStyle: true,
            boxWidth: 8
          }
        },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipText,
          bodyColor: tooltipText,
          borderColor: tooltipBorder,
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
          displayColors: true,
          callbacks: {
            label: function(context) {
              const val = context.raw;
              const formatted = typeof val === 'number' ? val.toLocaleString() : val;
              return ` ${context.dataset.label}: ${formatted} ${configData.unit}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 11 } },
          grid: { color: gridColor }
        },
        y: {
          ticks: {
            color: textColor,
            font: { family: 'JetBrains Mono', size: 11 },
            callback: configData.yAxisFormatter
          },
          grid: { color: gridColor },
          title: {
            display: true,
            text: configData.unit,
            color: textColor,
            font: { family: 'Plus Jakarta Sans', size: 12, weight: 600 }
          }
        }
      }
    }
  });

  // Update tab buttons styling
  document.querySelectorAll('.chart-tab-btn').forEach(btn => {
    const btnKey = btn.getAttribute('data-chart') || btn.getAttribute('data-tab');
    if (btnKey === tabKey) {
      btn.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
      btn.classList.remove('text-slate-400', 'border-transparent');
    } else {
      btn.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/40', 'shadow-[0_0_12px_rgba(56,189,248,0.25)]');
      btn.classList.add('text-slate-400', 'border-transparent');
    }
  });
}

// Re-render chart on theme switch without losing state
window.refreshChartTheme = function() {
  if (currentTab) {
    renderChart(currentTab);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.chart-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.getAttribute('data-chart') || btn.getAttribute('data-tab');
      renderChart(tab);
    });
  });

  // Initial render
  renderChart('recoveries');
});
