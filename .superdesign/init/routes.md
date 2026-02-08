# Routes

## Framework
- **Routing**: No router — static HTML files served via `python3 -m http.server 8000`
- **Navigation**: Direct `<a href>` links between files

## Route Map

| URL Path | File | Description |
|----------|------|-------------|
| `/landing.html` | `frontend/landing.html` | Marketing landing page |
| `/index.html` | `frontend/index.html` | Main dashboard application |

## Landing Page (`/landing.html`)
- Marketing/conversion page
- Sections: Hero, Problems, Features, How It Works, Integration/Code, CTA, Footer
- Links to `/index.html` via "Start Observing" and "Start Free" CTAs

## Dashboard (`/index.html`)
- Single-page app with 5 internal views managed via JS:
  1. **Journeys** — List of agent execution journeys
  2. **DAG Viewer** — Interactive DAG visualization for a selected journey
  3. **Optimizer** — Prompt optimization tool with Excel upload
  4. **Pipeline** — Semantic pipeline runner
  5. **Settings** — Configuration panel
- Internal navigation via sidebar buttons calling `showPage()` function
- No URL-based routing — all views rendered in same HTML file
