# Layouts

## Framework
- **Tech stack**: Vanilla HTML/CSS/JS (no framework, no build system)
- **Layout approach**: Two separate HTML files, each self-contained with inline styles

## Landing Page Layout (`frontend/landing.html`)
- Fixed top navigation bar with blur on scroll
- Full-width sections stacked vertically
- No sidebar
- Mobile hamburger menu overlay
- Footer at bottom

## Dashboard Layout (`frontend/index.html`)
- Sidebar (220px) + main content area
- Fixed header (52px) with breadcrumb navigation
- Sidebar navigation with collapsible sections
- Content area scrolls independently

## Navigation Component (Landing Page)
```css
.nav{
  position:fixed;
  top:0;
  left:0;
  right:0;
  z-index:1000;
  padding:0 24px;
  transition:background .3s ease, border-color .3s ease, backdrop-filter .3s ease;
  border-bottom:1px solid transparent;
}
.nav.scrolled{
  background:rgba(8,8,12,.85);
  backdrop-filter:blur(20px) saturate(1.2);
  -webkit-backdrop-filter:blur(20px) saturate(1.2);
  border-bottom-color:var(--border);
}
.nav-inner{
  max-width:1200px;
  margin:0 auto;
  height:64px;
  display:flex;
  align-items:center;
  justify-content:space-between;
}
```

```html
<nav class="nav" role="navigation" aria-label="Main navigation">
  <div class="nav-inner">
    <a href="/landing.html" class="nav-logo" aria-label="Arcana home">
      <div class="nav-logo-icon" aria-hidden="true"></div>
      <span class="nav-logo-text">Arcana</span>
    </a>
    <ul class="nav-links">
      <li><a href="#features">Features</a></li>
      <li><a href="#how-it-works">How It Works</a></li>
      <li><a href="#integration">Integration</a></li>
      <li><a href="https://github.com/arcana-dev/arcana" target="_blank" rel="noopener">Docs</a></li>
    </ul>
    <a href="/index.html" class="nav-cta desktop">Start Observing →</a>
    <button class="hamburger" aria-label="Toggle menu" aria-expanded="false">
      <span></span><span></span><span></span>
    </button>
  </div>
</nav>
```

## Sidebar Component (Dashboard)
```css
.sidebar{width:var(--sidebar-w);min-width:var(--sidebar-w);background:var(--sidebar);border-right:1px solid var(--border);display:flex;flex-direction:column;z-index:10;overflow-y:auto}
.sidebar-logo{padding:18px 20px 22px;border-bottom:1px solid var(--border)}
.sidebar-logo h1{font-family:var(--font-display);font-size:1.55rem;font-weight:400;letter-spacing:.02em;color:var(--text);line-height:1}
```

## Footer Component (Landing Page)
```css
.footer{
  border-top:1px solid var(--border);
  padding:40px 0;
  position:relative;
  z-index:1;
}
.footer-inner{
  display:flex;
  align-items:center;
  justify-content:space-between;
}
```

```html
<footer class="footer">
  <div class="container">
    <div class="footer-inner">
      <div class="footer-left">
        <span class="footer-logo">Arcana</span>
        <span class="footer-copy">© 2026 Arcana. All rights reserved.</span>
      </div>
      <ul class="footer-links">
        <li><a href="https://github.com/arcana-dev/arcana">GitHub</a></li>
        <li><a href="https://github.com/arcana-dev/arcana">Documentation</a></li>
        <li><a href="https://github.com/arcana-dev/arcana">API Reference</a></li>
      </ul>
    </div>
  </div>
</footer>
```
