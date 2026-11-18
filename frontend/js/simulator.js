/**
 * simulator.js
 * Live Interactive Concept Explorer / Epidemic SEIR-PINN Sandbox
 * Real-time Numerical ODE Simulation (120-Day Euler Integration)
 * Parameterized by Transmission (β), Recovery (γ), and Incubation (σ)
 */

(function initSimulator() {
  const sliderBeta = document.getElementById('slider-beta');
  const sliderGamma = document.getElementById('slider-gamma');
  const sliderSigma = document.getElementById('slider-sigma');
  const resetBtn = document.getElementById('sim-reset-btn');

  const valBeta = document.getElementById('val-beta');
  const valGamma = document.getElementById('val-gamma');
  const valSigma = document.getElementById('val-sigma');

  const r0ValEl = document.getElementById('sim-r0-val');
  const r0StatusEl = document.getElementById('sim-r0-status');
  const peakDayEl = document.getElementById('sim-peak-day');
  const peakInfectedEl = document.getElementById('sim-peak-infected');
  const finalRecoveredEl = document.getElementById('sim-final-recovered');
  const finalSusceptibleEl = document.getElementById('sim-final-susceptible');

  const canvas = document.getElementById('simulator-chart') || document.getElementById('sim-canvas');
  if (!canvas) {
    console.warn("Simulator canvas not found in DOM.");
    return;
  }
  const ctx = canvas.getContext('2d');

  let simChartInstance = null;

  // Defaults aligned with the paper's discovered PINN parameters
  const DEFAULTS = {
    beta: 0.45,
    gamma: 0.296,
    sigma: 0.19
  };

  function solveSEIR(beta, gamma, sigma, days = 120) {
    // Initial Conditions (Percentages)
    let S = 99.0;
    let E = 0.8;
    let I = 0.2;
    let R = 0.0;
    const N = 100.0;

    const dt = 0.2; // 5 sub-steps per day for numerical stability
    const subSteps = 5;

    const labels = [];
    const sData = [];
    const eData = [];
    const iData = [];
    const rData = [];

    let peakI = I;
    let peakDay = 0;

    for (let day = 0; day <= days; day++) {
      labels.push(`Day ${day}`);
      sData.push(parseFloat(S.toFixed(2)));
      eData.push(parseFloat(E.toFixed(2)));
      iData.push(parseFloat(I.toFixed(2)));
      rData.push(parseFloat(R.toFixed(2)));

      if (I > peakI) {
        peakI = I;
        peakDay = day;
      }

      // Euler sub-stepping for current day
      for (let step = 0; step < subSteps; step++) {
        const dS = - (beta * S * I) / N;
        const dE = ((beta * S * I) / N) - sigma * E;
        const dI = (sigma * E) - (gamma * I);
        const dR = gamma * I;

        S += dS * dt;
        E += dE * dt;
        I += dI * dt;
        R += dR * dt;

        // Conservation boundary clamps
        S = Math.max(0, Math.min(N, S));
        E = Math.max(0, Math.min(N, E));
        I = Math.max(0, Math.min(N, I));
        R = Math.max(0, Math.min(N, R));
      }
    }

    return {
      labels,
      sData,
      eData,
      iData,
      rData,
      peakI,
      peakDay,
      finalS: S,
      finalR: R
    };
  }

  function updateSimulation(animate = false) {
    const beta = parseFloat(sliderBeta?.value || DEFAULTS.beta);
    const gamma = parseFloat(sliderGamma?.value || DEFAULTS.gamma);
    const sigma = parseFloat(sliderSigma?.value || DEFAULTS.sigma);

    // Update readout labels
    if (valBeta) valBeta.textContent = beta.toFixed(3);
    if (valGamma) valGamma.textContent = gamma.toFixed(3);
    if (valSigma) valSigma.textContent = sigma.toFixed(3);

    // Basic Reproduction Number
    const R0 = beta / (gamma > 0 ? gamma : 0.001);
    if (r0ValEl) r0ValEl.textContent = R0.toFixed(2);

    if (r0StatusEl) {
      if (R0 > 1.05) {
        r0StatusEl.textContent = "Epidemic spreads sustained exponential expansion (R₀ > 1)";
        r0StatusEl.className = "text-xs text-rose-400 mt-1 font-mono-code";
      } else if (R0 >= 0.95) {
        r0StatusEl.textContent = "Epidemic is near critical endemic threshold (R₀ ≈ 1)";
        r0StatusEl.className = "text-xs text-amber-400 mt-1 font-mono-code";
      } else {
        r0StatusEl.textContent = "Epidemic contained, decaying towards extinction (R₀ < 1)";
        r0StatusEl.className = "text-xs text-emerald-400 mt-1 font-mono-code";
      }
    }

    // Run ODE Integration
    const sim = solveSEIR(beta, gamma, sigma, 120);

    // Update KPI Badges
    if (peakDayEl) peakDayEl.textContent = `Day ${sim.peakDay}`;
    if (peakInfectedEl) peakInfectedEl.textContent = `${sim.peakI.toFixed(1)}%`;
    if (finalRecoveredEl) finalRecoveredEl.textContent = `${sim.finalR.toFixed(1)}%`;
    if (finalSusceptibleEl) finalSusceptibleEl.textContent = `${sim.finalS.toFixed(1)}%`;

    // Render or update Chart.js
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const textColor = isLight ? '#334155' : '#94A3B8';
    const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)';

    // Downsample labels for cleaner x-axis (every 15 days)
    const displayLabels = sim.labels.map((l, idx) => idx % 15 === 0 || idx === 120 ? l : '');

    const chartConfig = {
      type: 'line',
      data: {
        labels: displayLabels,
        datasets: [
          {
            label: 'Infected I(t)',
            data: sim.iData,
            borderColor: '#FB7185',
            backgroundColor: 'rgba(251, 113, 133, 0.12)',
            borderWidth: 2.6,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 5
          },
          {
            label: 'Exposed E(t)',
            data: sim.eData,
            borderColor: '#FBBF24',
            backgroundColor: 'transparent',
            borderWidth: 2.0,
            borderDash: [4, 4],
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4
          },
          {
            label: 'Recovered R(t)',
            data: sim.rData,
            borderColor: '#34D399',
            backgroundColor: 'transparent',
            borderWidth: 2.2,
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 5
          },
          {
            label: 'Susceptible S(t)',
            data: sim.sData,
            borderColor: '#38BDF8',
            backgroundColor: 'transparent',
            borderWidth: 1.8,
            borderDash: [3, 3],
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: animate ? { duration: 600, easing: 'easeOutQuart' } : false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: textColor,
              font: { family: 'Plus Jakarta Sans', size: 11, weight: 600 },
              usePointStyle: true,
              boxWidth: 8
            }
          },
          tooltip: {
            backgroundColor: isLight ? '#FFFFFF' : '#0F111A',
            titleColor: isLight ? '#0F172A' : '#F8FAFC',
            bodyColor: isLight ? '#0F172A' : '#F8FAFC',
            borderColor: isLight ? '#CBD5E1' : '#334155',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              title: (ctx) => `Day ${ctx[0].dataIndex}`,
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw}% of Population`
            }
          }
        },
        scales: {
          x: {
            ticks: {
              color: textColor,
              font: { family: 'JetBrains Mono', size: 10 },
              maxRotation: 0,
              autoSkip: false
            },
            grid: { color: gridColor }
          },
          y: {
            min: 0,
            max: 100,
            ticks: {
              color: textColor,
              font: { family: 'JetBrains Mono', size: 10 },
              callback: (val) => `${val}%`
            },
            grid: { color: gridColor },
            title: {
              display: true,
              text: 'Population Share (%)',
              color: textColor,
              font: { family: 'Plus Jakarta Sans', size: 11, weight: 600 }
            }
          }
        }
      }
    };

    if (!simChartInstance) {
      simChartInstance = new Chart(ctx, chartConfig);
    } else {
      simChartInstance.data.labels = chartConfig.data.labels;
      simChartInstance.data.datasets = chartConfig.data.datasets;
      simChartInstance.options.scales.x.ticks.color = textColor;
      simChartInstance.options.scales.y.ticks.color = textColor;
      simChartInstance.options.plugins.legend.labels.color = textColor;
      simChartInstance.update(animate ? 'default' : 'none');
    }
  }

  // Event Listeners for real-time slider responsiveness
  [sliderBeta, sliderGamma, sliderSigma].forEach(slider => {
    slider?.addEventListener('input', () => updateSimulation(false));
  });

  // Reset button listener
  resetBtn?.addEventListener('click', () => {
    if (sliderBeta) sliderBeta.value = DEFAULTS.beta;
    if (sliderGamma) sliderGamma.value = DEFAULTS.gamma;
    if (sliderSigma) sliderSigma.value = DEFAULTS.sigma;
    updateSimulation(true);
  });

  // Theme switch hook
  window.addEventListener('resize', () => {
    if (simChartInstance) simChartInstance.resize();
  });

  // Expose theme refresh helper
  const originalThemeRefresh = window.refreshChartTheme;
  window.refreshChartTheme = function() {
    if (originalThemeRefresh) originalThemeRefresh();
    updateSimulation(false);
  };

  // Initial simulation calculation and render
  updateSimulation(true);
})();
