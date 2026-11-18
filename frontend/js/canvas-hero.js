/**
 * canvas-hero.js
 * Interactive Mathematical Epidemic Mesh & Particle Network
 * Stripe Press / Linear-grade canvas animation with spring physics,
 * harmonic wave interference, high-DPI scaling, and cursor interactivity.
 */

(function initHeroCanvas() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let width = 0;
  let height = 0;
  let dpr = window.devicePixelRatio || 1;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    width = canvas.parentElement ? canvas.parentElement.offsetWidth : window.innerWidth;
    height = canvas.parentElement ? canvas.parentElement.offsetHeight : window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);
  }

  resize();
  window.addEventListener('resize', resize);

  // Mouse interaction state
  const mouse = {
    x: width / 2,
    y: height / 2,
    targetX: width / 2,
    targetY: height / 2,
    radius: 170,
    isActive: false,
    clickRipples: []
  };

  window.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.targetX = e.clientX - rect.left;
    mouse.targetY = e.clientY - rect.top;
    mouse.isActive = true;
  });

  window.addEventListener('mouseleave', () => {
    mouse.isActive = false;
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    mouse.clickRipples.push({
      x: cx,
      y: cy,
      radius: 5,
      maxRadius: 240,
      opacity: 0.8,
      speed: 4.5
    });
  });

  // Particle Class
  const PARTICLE_COUNT = Math.min(Math.floor(window.innerWidth / 22), 65);
  const CONNECTION_DIST = 135;
  const particles = [];

  const colors = ['#38BDF8', '#818CF8', '#34D399'];

  class Particle {
    constructor(id) {
      this.id = id;
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.vx = (Math.random() - 0.5) * 0.65;
      this.vy = (Math.random() - 0.5) * 0.65;
      this.baseRadius = Math.random() * 1.8 + 1.2;
      this.radius = this.baseRadius;
      this.phase = Math.random() * Math.PI * 2;
      this.color = colors[id % colors.length];
    }

    update(time) {
      // Subtle harmonic oscillation
      this.x += this.vx + Math.sin(time * 0.0012 + this.phase) * 0.22;
      this.y += this.vy + Math.cos(time * 0.0012 + this.phase) * 0.22;

      // Wrap boundaries
      if (this.x < -15) this.x = width + 15;
      if (this.x > width + 15) this.x = -15;
      if (this.y < -15) this.y = height + 15;
      if (this.y > height + 15) this.y = -15;

      // Mouse repulsion
      if (mouse.isActive) {
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < mouse.radius && dist > 0) {
          const force = (1 - dist / mouse.radius) * 3.2;
          this.x += (dx / dist) * force;
          this.y += (dy / dist) * force;
          this.radius = this.baseRadius * 1.6;
        } else {
          this.radius = this.baseRadius;
        }
      }

      // Ripple shockwave influence
      for (let i = 0; i < mouse.clickRipples.length; i++) {
        const r = mouse.clickRipples[i];
        const rdx = this.x - r.x;
        const rdy = this.y - r.y;
        const rdist = Math.sqrt(rdx * rdx + rdy * rdy);
        const waveDist = Math.abs(rdist - r.radius);
        if (waveDist < 25 && rdist > 0) {
          const push = (1 - waveDist / 25) * 2.8 * r.opacity;
          this.x += (rdx / rdist) * push;
          this.y += (rdy / rdist) * push;
        }
      }
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.shadowBlur = 8;
      ctx.shadowColor = this.color;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push(new Particle(i));
  }

  // Draw Mathematical SEIR Wave Grids with Harmonic Multi-Tone
  function drawHarmonicWave(time) {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const waveColor1 = isLight ? 'rgba(2, 132, 199, 0.08)' : 'rgba(56, 189, 248, 0.05)';
    const waveColor2 = isLight ? 'rgba(99, 102, 241, 0.07)' : 'rgba(129, 140, 248, 0.04)';
    const waveColor3 = isLight ? 'rgba(5, 150, 105, 0.05)' : 'rgba(52, 211, 153, 0.03)';

    ctx.save();
    ctx.lineWidth = 1.4;

    // Primary SEIR Harmonic Curve
    ctx.strokeStyle = waveColor1;
    ctx.beginPath();
    for (let x = 0; x <= width; x += 12) {
      const y = height * 0.58 +
                Math.sin(x * 0.0032 + time * 0.0009) * 85 +
                Math.cos(x * 0.0068 + time * 0.0006) * 35;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Secondary Phase-Shifted Wave
    ctx.strokeStyle = waveColor2;
    ctx.beginPath();
    for (let x = 0; x <= width; x += 12) {
      const y = height * 0.68 +
                Math.cos(x * 0.0041 - time * 0.0008) * 75 +
                Math.sin(x * 0.0022 + time * 0.0004) * 45;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Tertiary Background Horizon Wave
    ctx.strokeStyle = waveColor3;
    ctx.beginPath();
    for (let x = 0; x <= width; x += 16) {
      const y = height * 0.46 +
                Math.sin(x * 0.0025 + time * 0.0005) * 60;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.restore();
  }

  // Animation Loop
  function animate(timestamp) {
    // Smooth cursor interpolation
    mouse.x += (mouse.targetX - mouse.x) * 0.08;
    mouse.y += (mouse.targetY - mouse.y) * 0.08;

    ctx.clearRect(0, 0, width, height);

    // Draw Mathematical Waveform
    drawHarmonicWave(timestamp);

    // Update and draw Ripples
    for (let i = mouse.clickRipples.length - 1; i >= 0; i--) {
      const rip = mouse.clickRipples[i];
      rip.radius += rip.speed;
      rip.opacity *= 0.96;

      ctx.save();
      ctx.beginPath();
      ctx.arc(rip.x, rip.y, rip.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(56, 189, 248, ${rip.opacity * 0.4})`;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();

      if (rip.radius > rip.maxRadius || rip.opacity < 0.02) {
        mouse.clickRipples.splice(i, 1);
      }
    }

    // Update and draw Particles
    for (let i = 0; i < particles.length; i++) {
      particles[i].update(timestamp);
      particles[i].draw();

      // Inter-particle connective web
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < CONNECTION_DIST) {
          const alpha = (1 - dist / CONNECTION_DIST) * 0.22;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
          ctx.lineWidth = 0.85;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
