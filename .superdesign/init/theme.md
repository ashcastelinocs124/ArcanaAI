# Theme / Design Tokens

## Framework
- **CSS approach**: Inline `<style>` blocks with CSS custom properties (`:root` variables)
- **No Tailwind** — pure vanilla CSS
- **Fonts**: Google Fonts (DM Sans + Instrument Serif)

## CSS Variables (Landing Page — `:root`)

```css
:root{
  --bg:#08080c;
  --card:#111118;
  --card-hover:#16161f;
  --border:#1e1e2c;
  --border-light:#2a2a3c;
  --copper:#c8956c;
  --copper-dim:rgba(200,149,108,.12);
  --copper-glow:rgba(200,149,108,.06);
  --green:#4ade80;
  --green-dim:rgba(74,222,128,.1);
  --red:#f87171;
  --red-dim:rgba(248,113,113,.1);
  --blue:#60a5fa;
  --blue-dim:rgba(96,165,250,.1);
  --purple:#a78bfa;
  --purple-dim:rgba(167,139,250,.1);
  --text:#e8e4df;
  --text-secondary:#7b7688;
  --text-dim:#4a4558;
  --font-body:'DM Sans',system-ui,sans-serif;
  --font-display:'Instrument Serif',Georgia,serif;
  --transition:all .2s cubic-bezier(.4,0,.2,1);
  --radius:8px;
  --radius-sm:5px;
  --radius-lg:12px;
}
```

## CSS Variables (Dashboard — `:root`)

```css
:root{
  --bg:#08080c;--sidebar:#0c0c12;--card:#111118;--card-hover:#16161f;
  --border:#1e1e2c;--border-light:#2a2a3c;
  --copper:#c8956c;--copper-dim:rgba(200,149,108,.12);--copper-glow:rgba(200,149,108,.06);
  --green:#4ade80;--green-dim:rgba(74,222,128,.1);
  --yellow:#fbbf24;--yellow-dim:rgba(251,191,36,.1);
  --red:#f87171;--red-dim:rgba(248,113,113,.1);
  --blue:#60a5fa;--blue-dim:rgba(96,165,250,.1);
  --purple:#a78bfa;--purple-dim:rgba(167,139,250,.1);
  --text:#e8e4df;--text-secondary:#7b7688;--text-dim:#4a4558;
  --sidebar-w:220px;--header-h:52px;
  --surface:#0e0e15;
  --radius:8px;--radius-sm:5px;--radius-lg:12px;
  --font-body:'DM Sans',system-ui,sans-serif;
  --font-display:'Instrument Serif',Georgia,serif;
  --transition:all .2s cubic-bezier(.4,0,.2,1);
}
```

## Fonts

```html
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
```

- **Body font**: DM Sans (weights: 300, 400, 500, 600)
- **Display font**: Instrument Serif (regular + italic)
- `font-display: swap` via Google Fonts default

## Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#08080c` | Page background (near-black) |
| `--sidebar` | `#0c0c12` | Dashboard sidebar background |
| `--card` | `#111118` | Card/panel backgrounds |
| `--card-hover` | `#16161f` | Card hover state |
| `--border` | `#1e1e2c` | Primary border color |
| `--border-light` | `#2a2a3c` | Secondary/lighter border |
| `--copper` | `#c8956c` | Primary accent (brand color) |
| `--copper-dim` | `rgba(200,149,108,.12)` | Copper background tint |
| `--copper-glow` | `rgba(200,149,108,.06)` | Copper subtle glow |
| `--green` | `#4ade80` | Success/pass state |
| `--red` | `#f87171` | Error/fail/drift state |
| `--blue` | `#60a5fa` | Info/neutral accent |
| `--purple` | `#a78bfa` | Secondary accent |
| `--yellow` | `#fbbf24` | Warning state (dashboard only) |
| `--text` | `#e8e4df` | Primary text (warm white) |
| `--text-secondary` | `#7b7688` | Secondary text (muted) |
| `--text-dim` | `#4a4558` | Dimmed text (labels) |

## Spacing & Sizing

| Token | Value | Usage |
|-------|-------|-------|
| `--radius` | `8px` | Default border radius |
| `--radius-sm` | `5px` | Small border radius (inputs) |
| `--radius-lg` | `12px` | Large border radius (cards) |
| `--sidebar-w` | `220px` | Dashboard sidebar width |
| `--header-h` | `52px` | Dashboard header height |
| Container | `max-width: 1200px` | Landing page content width |

## Background Atmosphere (Landing Page)

```css
body::before{
  content:'';
  position:fixed;
  inset:0;
  background:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(200,149,108,.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 60%, rgba(96,165,250,.03) 0%, transparent 50%),
    radial-gradient(ellipse 50% 50% at 20% 80%, rgba(167,139,250,.03) 0%, transparent 50%);
  pointer-events:none;
  z-index:0;
}

.grid-overlay{
  position:fixed;
  inset:0;
  background-image:
    radial-gradient(circle at 1px 1px, rgba(255,255,255,.03) 1px, transparent 0);
  background-size:40px 40px;
  pointer-events:none;
  z-index:0;
}

.scanline{
  position:fixed;
  inset:0;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(200,149,108,.008) 2px,
    rgba(200,149,108,.008) 4px
  );
  pointer-events:none;
  z-index:0;
}
```

## Breakpoints

| Breakpoint | Target |
|-----------|--------|
| `max-width: 767px` | Mobile |
| `max-width: 1023px` | Tablet |
| `min-width: 1024px` | Desktop |

## Transitions & Animations

- Default transition: `all .2s cubic-bezier(.4,0,.2,1)`
- Fade-in: `opacity .7s cubic-bezier(.16,1,.3,1), transform .7s cubic-bezier(.16,1,.3,1)`
- Float animations: 6-8s ease-in-out infinite (hero cards)
- Respects `prefers-reduced-motion: reduce`
