# Pages & Dependency Trees

## Framework
- **Tech stack**: Vanilla HTML/CSS/JS — single-file architecture
- **No imports**: All CSS and JS are inline within each HTML file
- **No component tree**: Everything is self-contained per file

## /landing.html (Landing Page)
Entry: `frontend/landing.html`
Dependencies: None (fully self-contained, ~1983 lines)

Sections rendered:
- Navigation (fixed top bar with blur)
- Hero (two-column: text + floating product mockup cards)
- Problem Statement (3-column grid of problem cards)
- Features (2x2 grid of feature cards)
- How It Works (3-step horizontal flow)
- Integration / Code Snippet (2-column: text + code block with typewriter)
- Final CTA (centered call-to-action)
- Footer

All CSS (~1200 lines) and JS (~370 lines) are inline in the file.

## /index.html (Dashboard App)
Entry: `frontend/index.html`
Dependencies:
- `frontend/data.json` (loaded via fetch at runtime)
- `https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js` (SheetJS for Excel parsing)

Internal views (all rendered by JS within the same file):
1. **Journeys** — Card grid of agent execution journeys
2. **DAG Viewer** — SVG-based DAG visualization with edge analysis
3. **Optimizer** — Excel upload + per-agent prompt optimization
4. **Pipeline** — Semantic pipeline runner with LLM integration
5. **Settings** — Configuration panel

All CSS (~500+ lines) and JS (~2000+ lines) are inline. File is ~2600+ lines total.
