/**
 * main.js
 * Master Application Controller:
 * - Scroll Progress & Glassmorphic Sticky Header
 * - Theme Switcher (Obsidian Cosmic Dark vs Crisp Paper Light)
 * - Spotlight Hover Gradient Tracking (Linear / Vercel style)
 * - 3D Card Tilt Micro-Interactions
 * - Scroll-Triggered Stagger Reveals
 * - Viewport-Triggered Repeated Counter Animations (R² = 0.976, etc.)
 * - Interactive BibTeX Copy with Toast
 * - Mobile Navigation Menu
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── 1. Reading Progress Bar & Sticky Header ───────────────────────────────
  const progressBar = document.getElementById('reading-progress');
  const navbar = document.getElementById('main-navbar');

  window.addEventListener('scroll', () => {
    const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = totalHeight > 0 ? (window.scrollY / totalHeight) * 100 : 0;
    if (progressBar) progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;

    if (navbar) {
      if (window.scrollY > 40) {
        navbar.classList.add('shadow-xl', 'border-b', 'border-slate-800/90', 'bg-[#090A0F]/90');
      } else {
        navbar.classList.remove('shadow-xl', 'border-b', 'border-slate-800/90', 'bg-[#090A0F]/90');
      }
    }
  });

  // ── 2. Theme Toggle Controller ───────────────────────────────────────────
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const themeIconDark = document.getElementById('theme-icon-dark');
  const themeIconLight = document.getElementById('theme-icon-light');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('academic-showcase-theme', theme);
    if (theme === 'light') {
      themeIconDark?.classList.add('hidden');
      themeIconLight?.classList.remove('hidden');
    } else {
      themeIconDark?.classList.remove('hidden');
      themeIconLight?.classList.add('hidden');
    }
    // Notify charts to update colors
    if (window.refreshChartTheme) {
      window.refreshChartTheme();
    }
  }

  // Load saved or system preference
  const savedTheme = localStorage.getItem('academic-showcase-theme') || 
    (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(savedTheme);

  themeToggleBtn?.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'light' ? 'dark' : 'light');
  });

  // ── 3. Interactive Cursor Spotlight on Cards (Linear / Vercel style) ──────
  const spotlightCards = document.querySelectorAll('.spotlight-card, .bento-card, .glass-panel');
  spotlightCards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty('--mouse-x', `${x}px`);
      card.style.setProperty('--mouse-y', `${y}px`);
    });
  });

  // ── 4. Subtle 3D Perspective Tilt on Bento & Feature Cards ────────────────
  const tiltCards = document.querySelectorAll('.tilt-card, .bento-card');
  tiltCards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -5; // max 5 deg tilt
      const rotateY = ((x - centerX) / centerX) * 5;

      card.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) scale3d(1.015, 1.015, 1.015)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
    });
  });

  // ── 5. Scroll-Triggered Stagger Reveal System ──────────────────────────────
  const revealElements = document.querySelectorAll('.reveal-on-scroll');
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-revealed');
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px'
  });

  revealElements.forEach(el => revealObserver.observe(el));

  // ── 6. Number Counter Animation on Scroll (Repeats Every Time in View) ───
  function resetCounter(obj) {
    if (obj._animationFrameId) {
      cancelAnimationFrame(obj._animationFrameId);
      obj._animationFrameId = null;
    }
    const rawTarget = obj.getAttribute('data-counter-target') || '0';
    const isDecimal = rawTarget.includes('.');
    const suffix = obj.getAttribute('data-counter-suffix') || '';
    const prefix = obj.getAttribute('data-counter-prefix') || '';
    obj.textContent = `${prefix}${isDecimal ? '0.000' : '0'}${suffix}`;
  }

  function animateValue(obj) {
    if (obj._animationFrameId) {
      cancelAnimationFrame(obj._animationFrameId);
      obj._animationFrameId = null;
    }

    const rawTarget = obj.getAttribute('data-counter-target') || '0';
    const isDecimal = rawTarget.includes('.');
    const target = parseFloat(rawTarget);
    const suffix = obj.getAttribute('data-counter-suffix') || '';
    const prefix = obj.getAttribute('data-counter-prefix') || '';
    const duration = 1500;
    const startTime = performance.now();

    function updateCounter(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Smooth cubic ease-out
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = progress === 1 ? target : (target * easeOut);

      obj.textContent = `${prefix}${isDecimal ? current.toFixed(3) : Math.round(current).toLocaleString()}${suffix}`;

      if (progress < 1) {
        obj._animationFrameId = requestAnimationFrame(updateCounter);
      } else {
        obj._animationFrameId = null;
      }
    }

    obj._animationFrameId = requestAnimationFrame(updateCounter);
  }

  // Observe all counter elements with enter/exit triggers
  const countElements = document.querySelectorAll('[data-counter-target]');
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const target = entry.target;
      if (entry.isIntersecting) {
        animateValue(target);
      } else {
        resetCounter(target);
      }
    });
  }, {
    threshold: 0.15,
    rootMargin: '0px 0px -30px 0px'
  });

  countElements.forEach(el => counterObserver.observe(el));

  // ── 7. BibTeX Copy to Clipboard & Toast ────────────────────────────────────
  const copyBibtexBtn = document.getElementById('copy-bibtex-btn');
  const copyToast = document.getElementById('copy-toast');
  const bibtexContent = document.getElementById('bibtex-code');

  copyBibtexBtn?.addEventListener('click', () => {
    if (!bibtexContent) return;
    const textToCopy = bibtexContent.innerText.trim();

    navigator.clipboard.writeText(textToCopy).then(() => {
      // Toggle button visual state
      const btnText = copyBibtexBtn.querySelector('.btn-copy-text');
      const iconCheck = copyBibtexBtn.querySelector('.icon-check');
      const iconCopy = copyBibtexBtn.querySelector('.icon-copy');

      if (btnText) btnText.textContent = "Copied to Clipboard!";
      if (iconCheck) iconCheck.classList.remove('hidden');
      if (iconCopy) iconCopy.classList.add('hidden');

      // Show toast
      if (copyToast) {
        copyToast.classList.remove('opacity-0', 'pointer-events-none', 'translate-y-4');
        copyToast.classList.add('opacity-100', 'translate-y-0');

        setTimeout(() => {
          copyToast.classList.add('opacity-0', 'pointer-events-none', 'translate-y-4');
          copyToast.classList.remove('opacity-100', 'translate-y-0');
        }, 3200);
      }

      setTimeout(() => {
        if (btnText) btnText.textContent = "Copy BibTeX";
        if (iconCheck) iconCheck.classList.add('hidden');
        if (iconCopy) iconCopy.classList.remove('hidden');
      }, 2500);
    });
  });

  // ── 8. Mobile Navigation Drawer ───────────────────────────────────────────
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const mobileMenu = document.getElementById('mobile-menu');

  mobileMenuBtn?.addEventListener('click', () => {
    if (mobileMenu) {
      mobileMenu.classList.toggle('hidden');
    }
  });

  document.querySelectorAll('.mobile-nav-link').forEach(link => {
    link.addEventListener('click', () => {
      if (mobileMenu) mobileMenu.classList.add('hidden');
    });
  });
});
