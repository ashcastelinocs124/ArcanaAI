# Arcana Frontend Reference

**Purpose:** Reliably modify the Arcana single-file frontend (`frontend/index.html`) — a ~2300-line HTML file with inline CSS and JS. All state, rendering, navigation, and API calls live in one file.

## Architecture Quick Reference

### File Structure (single file)
```
frontend/index.html
├── <head>        CDN scripts, <style> block (lines 1-250)
├── <body>        Sidebar, header, page containers (lines 251-365)
└── <script>      All JS: state, navigation, rendering, API calls (lines 366-end)
```

### State Pattern
```javascript
let DATA = { journeys: [] };          // Main data (loaded via upload)
let dataLoaded = false;                // Controls empty-state rendering
let currentPage = 'overview';          // Active page key
let optimizerState = { ... };          // Prompt optimizer state
// Additional feature state goes here (e.g., promptLibraryCompare)
```

### Navigation System (5 registration points)
Adding a new page requires changes in ALL of these:
1. **Sidebar item**: `<div class="sidebar-item" data-page="KEY">` (in HTML body)
2. **Page container**: `<div class="page" id="page-KEY"></div>` (in HTML body)
3. **`renderCurrentPage()`**: Add `else if (page === 'KEY') renderFunction();`
4. **`renderPage()` switch**: Add `case 'KEY': renderFunction(); break;`
5. **`navigateTo()` names**: Add `KEY:'Display Name'` to the names object

Missing any of these causes silent navigation failures.

### Rendering Pattern
Every page uses the same pattern:
```javascript
function renderMyPage() {
  if (!dataLoaded) return renderEmptyState('page-KEY', 'Title', 'Description');
  document.getElementById('page-KEY').innerHTML = `...template literal...`;
}
```

### File Upload Pattern (DOM re-render trap)
**CRITICAL BUG PATTERN:** When a file upload handler calls a render function, the DOM is recreated, destroying `<input type="file">` elements and their `FileList`. Always store the `File` object in state:
```javascript
// WRONG: DOM input destroyed on re-render, file reference lost
optimizerState.fileName = file.name;
renderOptimizer(); // Recreates DOM — file input is now empty

// CORRECT: Store File object in state, read from state later
optimizerState.file = file;
optimizerState.fileName = file.name;
renderOptimizer();
// In runOptimizer(): use optimizerState.file, NOT document.getElementById('input').files
```

### Excel/XLSX Parsing Pattern
Uses SheetJS CDN (`xlsx.full.min.js`). Column names from real Excel files vary wildly (`Input`, `input_text`, `INPUT`, etc.). Always use fuzzy column resolution:
```javascript
function resolveColumn(headers, target) {
  var tLower = target.toLowerCase().replace(/[\s_-]/g, '');
  for (var i = 0; i < headers.length; i++) {
    var hLower = headers[i].toLowerCase().replace(/[\s_-]/g, '');
    if (hLower === tLower) return headers[i];
  }
  for (var i = 0; i < headers.length; i++) {
    var hLower = headers[i].toLowerCase().replace(/[\s_-]/g, '');
    if (hLower.indexOf(tLower) !== -1 || tLower.indexOf(hLower) !== -1) return headers[i];
  }
  return null;
}
```
Never access `row.input` directly — use `row[resolvedColumnName]`.

### Backend Integration Pattern
Always check backend health first with a short timeout, then fall back to client-side:
```javascript
var backendAvailable = false;
try {
  var resp = await fetch('http://localhost:5000/health', { signal: AbortSignal.timeout(3000) });
  backendAvailable = resp.ok;
} catch (e) { backendAvailable = false; }

if (backendAvailable) {
  // Send to backend API
} else {
  // Client-side fallback
}
```

### localStorage Persistence Pattern
For user data that should survive page refresh:
```javascript
let myState = {
  apiKey: localStorage.getItem('arcana-my-key') || '',
};
// In UI: onchange="myState.val=this.value;localStorage.setItem('arcana-my-key',this.value)"
```

### CSS Variables (design system)
| Variable | Value | Use |
|----------|-------|-----|
| `--copper` | `#c8956c` | Primary accent, active states |
| `--green` | `#4ade80` | Success, pass, healthy |
| `--red` | `#f87171` | Error, fail, issues |
| `--blue` | `#60a5fa` | Info, models, latency |
| `--card` | `#111118` | Card backgrounds |
| `--bg` | `#08080c` | Page background |
| `--border` | `#1e1e2c` | Default borders |
| `--font-display` | `Instrument Serif` | Headings, large numbers |
| `--font-body` | `DM Sans` | Body text |

### Existing Pages
| Page Key | Function | Section |
|----------|----------|---------|
| `overview` | `renderOverview()` | Project |
| `traces` | `renderTraces()` | Monitoring |
| `metrics` | `renderMetrics()` | Monitoring |
| `logs` | `renderLogs()` | Monitoring |
| `pipeline` | `renderPipeline()` | Analysis |
| `optimizer` | `renderOptimizer()` | Analysis |
| `dagviewer` | `renderDAGViewer()` | Workbench |
| `evaluations` | `renderEvaluations()` | Workbench |
| `promptlib` | `renderPromptLibrary()` | Workbench |
| `settings` | `renderSettings()` | Operations |

## Execution Workflow

### Phase 1: Read Before Edit
1. Read the target section of `frontend/index.html` using offset/limit (file is ~2300 lines)
2. Identify the exact insertion points for each change
3. Map all 5 navigation registration points if adding a page

### Phase 2: Make Changes (order matters)
1. CDN scripts in `<head>` section
2. Sidebar items in HTML body (maintain section grouping)
3. Page containers after existing `<div class="page">` elements
4. State variables after existing state declarations
5. Helper functions grouped near related features
6. Render functions before the INIT section at bottom
7. Navigation registration at all 5 points
8. `window.xxx = xxx` exports for onclick handlers

### Phase 3: Verify
```bash
sed -n '/<script>/,/<\/script>/p' frontend/index.html | sed '1d;$d' > /tmp/check.js && node --check /tmp/check.js
```

## Known Bug Patterns

### 1. File reference lost after re-render
Store `File` objects in JS state, not in DOM inputs. Re-rendering destroys `<input type="file">` elements.

### 2. Excel column name mismatch
Never hardcode `row.input` or `row.gold`. Use `resolveColumn()` for case-insensitive fuzzy matching.

### 3. `st.` alias scoping
`const st = optimizerState` is defined inside `renderOptimizer()`. Other functions like `runOptimizer()` must use `optimizerState.` directly.

### 4. Missing `window.` export
Functions called from HTML `onclick` attributes must be exported: `window.myFunc = myFunc`.

## Quality Guidelines

**ALWAYS:**
- Read the file section before editing (use offset/limit for large reads)
- Store File objects in state, never rely on DOM file inputs surviving re-renders
- Use fuzzy column resolution for any Excel/CSV column access
- Register new pages in ALL 5 navigation points
- Run `node --check` after changes
- Use `var` in functions that use `.forEach` callbacks (avoid `let` scoping issues in older patterns)
- Export onclick handlers: `window.myFunc = myFunc`

**NEVER:**
- Access Excel row properties directly (`row.input`) — always resolve column names first
- Re-query DOM for file inputs after calling a render function
- Hardcode column names when parsing user-uploaded data
- Forget the `window.xxx` export for functions called from HTML onclick attributes
- Skip the syntax check — template literal bugs are invisible until runtime
- Use `st.` (local alias) outside of the function where it's defined
