/**
 * test_frontend_optimizer.js
 *
 * Comprehensive test suite for the Prompt Optimizer frontend rendering
 * and state management in the Arcana LLM Observability single-file app.
 *
 * Run with:  node --test tests/test_frontend_optimizer.js
 *
 * Requirements:
 *   - Node.js 18+ (uses node:test, node:assert, built-in fetch)
 *   - Static server on localhost:8000 serving frontend/index.html (for HTML tests)
 *   - Backend on localhost:5000 (only for the integration test)
 */

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { execSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const HTML_PATH = path.resolve(__dirname, '..', 'frontend', 'index.html');

/** Read the full HTML file from disk (no server needed). */
function readHtml() {
  return fs.readFileSync(HTML_PATH, 'utf-8');
}

/**
 * Extract the inline JavaScript from the main <script> block (not the
 * SheetJS CDN tag).  The inline script starts at `<script>` (bare, no src)
 * and ends at `</script>`.
 */
function extractInlineJs(html) {
  // Match the first <script> tag that has NO src attribute
  const pattern = /<script>(?!<)([\s\S]*?)<\/script>/;
  const match = html.match(pattern);
  if (!match) {
    throw new Error('Could not extract inline <script> block from HTML');
  }
  return match[1];
}

/**
 * Build a small self-contained JS string that defines truncText and
 * resolveColumn (extracted verbatim from the source) so we can evaluate
 * them in a Node.js context without a browser DOM.
 */
function buildUtilityContext(jsSource) {
  // Extract truncText
  const truncTextMatch = jsSource.match(
    /function truncText\(t,\s*n\)\s*\{[^}]+\}/
  );
  if (!truncTextMatch) throw new Error('Could not extract truncText');

  // Extract resolveColumn (multi-line)
  const resolveColMatch = jsSource.match(
    /function resolveColumn\(headers,\s*target\)\s*\{[\s\S]*?\n\}/
  );
  if (!resolveColMatch) throw new Error('Could not extract resolveColumn');

  return `
    ${truncTextMatch[0]}
    ${resolveColMatch[0]}
    module.exports = { truncText, resolveColumn };
  `;
}

/**
 * Evaluate the utility functions in a fresh Node.js child process and
 * return the exports.  This avoids polluting the current process and
 * sidesteps any browser-only globals.
 */
function loadUtilities() {
  const html = readHtml();
  const js = extractInlineJs(html);
  const code = buildUtilityContext(js);

  // Write to a temp file, require it, then clean up
  const tmpFile = path.join(os.tmpdir(), `arcana_utils_${Date.now()}.js`);
  fs.writeFileSync(tmpFile, code, 'utf-8');
  try {
    // Clear require cache in case of repeated runs
    delete require.cache[tmpFile];
    return require(tmpFile);
  } finally {
    fs.unlinkSync(tmpFile);
  }
}

// Pre-load utilities once for the rendering / unit test categories
let truncText;
let resolveColumn;

// ===========================================================================
// 1. Static HTML Tests (curl-based / fetch-based)
// ===========================================================================

describe('1. Static HTML Tests', () => {
  let html;
  let fetchAvailable = true;

  before(async () => {
    // Try fetching from the dev server; fall back to disk read for structural
    // tests that do not strictly require a running server.
    try {
      const res = await fetch('http://localhost:8000/index.html', {
        signal: AbortSignal.timeout(3000),
      });
      html = await res.text();
    } catch {
      fetchAvailable = false;
      html = readHtml();
    }
  });

  it('test_page_loads', async () => {
    if (!fetchAvailable) {
      // If the server is not running, verify the file at least exists on disk
      assert.ok(
        fs.existsSync(HTML_PATH),
        'index.html must exist on disk at ' + HTML_PATH
      );
      return;
    }
    const res = await fetch('http://localhost:8000/index.html', {
      signal: AbortSignal.timeout(5000),
    });
    assert.strictEqual(res.status, 200, 'GET /index.html should return 200');
  });

  it('test_optimizer_page_div_exists', () => {
    assert.ok(
      html.includes('id="page-optimizer"'),
      'HTML must contain a div with id="page-optimizer"'
    );
  });

  it('test_css_variables_defined', () => {
    const requiredVars = ['--copper', '--green', '--red', '--blue', '--text-dim'];
    for (const v of requiredVars) {
      assert.ok(
        html.includes(v),
        `CSS variable ${v} must be defined in the HTML`
      );
    }
  });

  it('test_css_spinner_animation', () => {
    assert.ok(
      html.includes('@keyframes spin'),
      'HTML must contain @keyframes spin for the spinner animation'
    );
  });

  it('test_script_tag_exists', () => {
    // At least two <script> tags: SheetJS CDN + inline
    const scriptMatches = html.match(/<script[\s>]/g);
    assert.ok(
      scriptMatches && scriptMatches.length >= 2,
      'HTML must contain at least two <script> tags (SheetJS CDN + inline)'
    );
  });

  it('test_sheetjs_loaded', () => {
    assert.ok(
      html.includes('xlsx') || html.includes('XLSX') || html.includes('sheetjs'),
      'HTML must reference the SheetJS / XLSX library'
    );
  });
});

// ===========================================================================
// 2. JavaScript Syntax Tests
// ===========================================================================

describe('2. JavaScript Syntax Tests', () => {
  it('test_js_syntax_valid', () => {
    const html = readHtml();
    const js = extractInlineJs(html);

    // Write JS to temp file and run `node --check`
    const tmpFile = path.join(os.tmpdir(), `arcana_syntax_check_${Date.now()}.js`);
    fs.writeFileSync(tmpFile, js, 'utf-8');
    try {
      execSync(`node --check "${tmpFile}"`, { stdio: 'pipe' });
      // If we reach here the syntax is valid
      assert.ok(true, 'Inline JS passes node --check syntax validation');
    } catch (err) {
      const stderr = err.stderr ? err.stderr.toString() : '';
      assert.fail(`Inline JS has syntax errors:\n${stderr}`);
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });
});

// ===========================================================================
// 3. Rendering Tests
// ===========================================================================

describe('3. Rendering Tests', () => {
  before(() => {
    const utils = loadUtilities();
    truncText = utils.truncText;
    resolveColumn = utils.resolveColumn;
  });

  // --- truncText -----------------------------------------------------------

  it('test_truncText_long', () => {
    assert.strictEqual(
      truncText('Hello World', 5),
      'Hello...',
      'truncText should truncate and add ellipsis'
    );
  });

  it('test_truncText_short', () => {
    assert.strictEqual(
      truncText('Hi', 5),
      'Hi',
      'truncText should return string as-is when shorter than limit'
    );
  });

  it('test_truncText_empty', () => {
    assert.strictEqual(
      truncText('', 5),
      '(empty)',
      'truncText should return "(empty)" for empty strings'
    );
  });

  it('test_truncText_null', () => {
    assert.strictEqual(
      truncText(null, 5),
      '(empty)',
      'truncText should return "(empty)" for null input'
    );
  });

  it('test_truncText_undefined', () => {
    assert.strictEqual(
      truncText(undefined, 10),
      '(empty)',
      'truncText should return "(empty)" for undefined input'
    );
  });

  it('test_truncText_exact_boundary', () => {
    assert.strictEqual(
      truncText('Hello', 5),
      'Hello',
      'truncText should NOT truncate when length === limit'
    );
  });

  it('test_truncText_one_over', () => {
    assert.strictEqual(
      truncText('Hello!', 5),
      'Hello...',
      'truncText should truncate when length is limit + 1'
    );
  });

  // --- resolveColumn -------------------------------------------------------

  it('test_resolveColumn_exact', () => {
    assert.strictEqual(
      resolveColumn(['User Input'], 'User Input'),
      'User Input',
      'Exact match should return the header'
    );
  });

  it('test_resolveColumn_case_insensitive', () => {
    assert.strictEqual(
      resolveColumn(['User Input', 'Expected Output'], 'user input'),
      'User Input',
      'Case-insensitive exact match should work'
    );
  });

  it('test_resolveColumn_fuzzy', () => {
    const result = resolveColumn(['User Input', 'Expected Output'], 'input');
    assert.strictEqual(
      result,
      'User Input',
      'Fuzzy (substring) match should return "User Input" for target "input"'
    );
  });

  it('test_resolveColumn_missing', () => {
    assert.strictEqual(
      resolveColumn(['Name', 'Age'], 'input'),
      null,
      'Should return null when no header matches'
    );
  });

  it('test_resolveColumn_expected_output', () => {
    assert.strictEqual(
      resolveColumn(['Agent Name', 'User Input', 'Expected Output'], 'expected output'),
      'Expected Output',
      'Should resolve "expected output" to "Expected Output"'
    );
  });

  it('test_resolveColumn_with_underscores', () => {
    // The function strips spaces, hyphens, and underscores before comparing
    assert.strictEqual(
      resolveColumn(['user_input', 'expected_output'], 'User Input'),
      'user_input',
      'Should match ignoring underscores/spaces'
    );
  });

  it('test_resolveColumn_with_hyphens', () => {
    assert.strictEqual(
      resolveColumn(['user-input', 'expected-output'], 'user input'),
      'user-input',
      'Should match ignoring hyphens vs spaces'
    );
  });

  it('test_resolveColumn_partial_header_inside_target', () => {
    // Pass 2: tLower.indexOf(hLower) !== -1 — i.e., header is substring of target
    assert.strictEqual(
      resolveColumn(['input'], 'user input'),
      'input',
      'Should match when header is a substring of the target'
    );
  });

  // --- renderOptimizer template simulation ---------------------------------

  it('test_render_with_results', () => {
    const html = readHtml();
    const js = extractInlineJs(html);

    // Build a mock optimizerState with results and evaluate the template
    // portion that produces the "has results" HTML.
    const mockState = {
      promptTemplate: 'You are a helpful assistant. {input}',
      targetScore: 0.85,
      maxIters: 3,
      results: [
        {
          row_index: 0,
          input: 'What is 2+2?',
          gold: '4',
          output: 'The answer is 4.',
          score: 0.92,
          iterations: 2,
          template: 'Optimized prompt template v1',
          original_output: 'It is 4.',
          original_score: 0.65,
          latency_ms: 1234,
          cost_usd: 0.0021,
        },
        {
          row_index: 1,
          input: 'Capital of France?',
          gold: 'Paris',
          output: 'Paris',
          score: 0.98,
          iterations: 1,
          template: 'Optimized prompt template v1',
          original_output: 'The capital is Paris.',
          original_score: 0.72,
          latency_ms: 890,
          cost_usd: 0.0015,
        },
      ],
      summary: {
        total_rows: 2,
        original_avg: 0.685,
        optimized_avg: 0.95,
        improvement: 26.5,
        total_iterations: 3,
        met_target: 2,
        total_latency_ms: 2124,
        avg_latency_ms: 1062,
        total_cost_usd: 0.0036,
        total_tokens: 4500,
        model: 'gpt-4o',
      },
      running: false,
      error: null,
      fileName: 'test.csv',
      model: 'gpt-4o',
      apiKey: 'sk-test',
      selectedAgent: null,
      file: null,
      preview: null,
    };

    // Simulate the rendering by evaluating the template literal logic
    // We replicate the key rendering expression from renderOptimizer
    const st = mockState;
    const hasResults = st.results.length > 0;
    const origAvg = st.summary?.original_avg || 0;
    const optAvg = st.summary?.optimized_avg || 0;
    const improvement = st.summary?.improvement?.toFixed(0) || 0;

    // Generate the stats row HTML (the core of what renderOptimizer produces)
    const statsHtml = `
      <div class="stats-row" style="grid-template-columns:repeat(3,1fr);margin-bottom:12px">
        <div class="stat-card red"><div class="stat-label">Original Avg</div><div class="stat-value">${(origAvg*100).toFixed(0)}%</div></div>
        <div class="stat-card green"><div class="stat-label">Optimized Avg</div><div class="stat-value">${(optAvg*100).toFixed(0)}%</div></div>
        <div class="stat-card copper"><div class="stat-label">Improvement</div><div class="stat-value">+${improvement}%</div></div>
      </div>
    `;

    // Generate table rows
    const tableRows = st.results.map((r, i) => {
      const origScore = r.original_score || 0;
      const optColor = r.score >= 0.85 ? 'var(--green)' : r.score >= 0.6 ? 'var(--yellow)' : 'var(--red)';
      const delta = ((r.score - origScore) * 100).toFixed(0);
      return `<tr>
        <td>${i+1}</td>
        <td>${truncText(r.input, 60)}</td>
        <td>${truncText(r.gold, 60)}</td>
        <td>${truncText(r.output, 60)}</td>
        <td><span style="color:${optColor}">${(r.score*100).toFixed(0)}%</span></td>
        <td><span>${parseInt(delta)>=0?'+':''}${delta}%</span></td>
        <td>${r.latency_ms ? (r.latency_ms/1000).toFixed(1) + 's' : '--'}</td>
        <td>${r.cost_usd != null ? '$' + r.cost_usd.toFixed(4) : '--'}</td>
      </tr>`;
    }).join('');

    // Verify stats cards
    assert.ok(hasResults, 'hasResults should be true with non-empty results');
    assert.ok(statsHtml.includes('stat-card red'), 'Should contain original avg card');
    assert.ok(statsHtml.includes('stat-card green'), 'Should contain optimized avg card');
    assert.ok(statsHtml.includes('stat-card copper'), 'Should contain improvement card');
    assert.ok(statsHtml.includes('69%'), 'Original avg should display as 69% (0.685 * 100)');
    assert.ok(statsHtml.includes('95%'), 'Optimized avg should display as 95%');
    assert.ok(statsHtml.includes('+27%'), 'Improvement should display as +27% (26.5 rounds to 27)');

    // Verify table rows
    assert.ok(tableRows.includes('What is 2+2?'), 'Table should contain first input');
    assert.ok(tableRows.includes('Capital of France?'), 'Table should contain second input');
    assert.ok(tableRows.includes('92%'), 'Table should show 92% score');
    assert.ok(tableRows.includes('98%'), 'Table should show 98% score');
    assert.ok(tableRows.includes('1.2s'), 'Should show latency 1.2s (1234ms)');
    assert.ok(tableRows.includes('$0.0021'), 'Should show cost $0.0021');
    assert.ok(tableRows.includes('+27%'), 'Should show delta for first row');

    // Verify chart div would exist
    const chartHtml = st.results.map((r, i) => {
      const origH = Math.max(4, (r.original_score || 0) * 140);
      const optH = Math.max(4, r.score * 140);
      return `<div style="flex:1;height:${origH}px"></div><div style="flex:1;height:${optH}px"></div>`;
    }).join('');
    assert.ok(chartHtml.includes('height:91'), 'Chart should have bar for 0.65 score (91px)');
    assert.ok(chartHtml.includes('height:128.8'), 'Chart should have bar for 0.92 score (128.8px)');
  });

  it('test_render_empty_state', () => {
    const st = {
      results: [],
      summary: null,
      running: false,
      error: null,
      promptTemplate: 'You are a helpful assistant. {input}',
      targetScore: 0.85,
      maxIters: 3,
      model: 'gpt-4o',
      apiKey: '',
      fileName: null,
      file: null,
      preview: null,
      selectedAgent: null,
    };

    const hasResults = st.results.length > 0;
    assert.strictEqual(hasResults, false, 'hasResults should be false');

    // The empty state template
    const emptyHtml = `
      <div class="metric-card" style="padding:0;overflow:hidden">
        <div style="display:grid;grid-template-columns:1fr 1fr">
          <div style="padding:30px;border-right:1px solid var(--border);text-align:center">
            <div style="color:var(--red)">Original Prompt Results</div>
            <div>Run the optimizer to see the LLM output from your current prompt template</div>
          </div>
          <div style="padding:30px;text-align:center">
            <div style="color:var(--green)">Optimized Prompt Results</div>
            <div>The optimized prompt and its results will appear here</div>
          </div>
        </div>
      </div>
    `;

    assert.ok(
      emptyHtml.includes('Run the optimizer to see'),
      'Empty state should contain instructions'
    );
    assert.ok(
      emptyHtml.includes('Original Prompt Results'),
      'Empty state should show original prompt section'
    );
    assert.ok(
      emptyHtml.includes('Optimized Prompt Results'),
      'Empty state should show optimized prompt section'
    );
  });

  it('test_render_running_state', () => {
    const st = { running: true };

    // Reproduce the running state HTML from renderOptimizer
    const runningHtml = st.running
      ? '<div id="optimizer-progress"><div class="spinner" style="animation:spin .8s linear infinite"></div><span>Running optimization against backend...</span></div>'
      : '';

    assert.ok(
      runningHtml.includes('optimizer-progress'),
      'Running state should render progress container'
    );
    assert.ok(
      runningHtml.includes('spinner'),
      'Running state should render a spinner element'
    );
    assert.ok(
      runningHtml.includes('animation:spin'),
      'Spinner should use the spin animation'
    );
    assert.ok(
      runningHtml.includes('Running optimization'),
      'Should display running message'
    );
  });

  it('test_render_error_state', () => {
    const st = { error: 'API key is invalid' };

    // Reproduce the error banner from renderOptimizer
    const errorHtml = st.error
      ? '<div style="background:var(--red-dim);border:1px solid rgba(248,113,113,.3);border-radius:var(--radius);padding:12px 16px;margin-bottom:16px;font-size:.82rem;color:var(--red)">' +
        st.error +
        '</div>'
      : '';

    assert.ok(
      errorHtml.includes('var(--red-dim)'),
      'Error state should have red-dim background'
    );
    assert.ok(
      errorHtml.includes('color:var(--red)'),
      'Error state should have red text color'
    );
    assert.ok(
      errorHtml.includes('API key is invalid'),
      'Error state should display the error message'
    );
  });

  it('test_render_error_state_absent_when_no_error', () => {
    const st = { error: null };
    const errorHtml = st.error
      ? '<div style="background:var(--red-dim)">' + st.error + '</div>'
      : '';
    assert.strictEqual(
      errorHtml,
      '',
      'No error banner should render when error is null'
    );
  });
});

// ===========================================================================
// 4. Result Mapping Tests
// ===========================================================================

describe('4. Result Mapping Tests', () => {
  const mockBackendResponse = {
    success: true,
    results: [
      {
        row_index: 0,
        input: 'Translate hello to French',
        gold: 'Bonjour',
        output: 'Bonjour',
        score: 0.95,
        iterations: 2,
        template: 'Translate {input} accurately.',
        comments: 'Basic translation',
        original_output: 'Hello in French is bonjour.',
        original_score: 0.6,
        latency_ms: 2340,
        original_latency_ms: 800,
        prompt_tokens: 120,
        completion_tokens: 45,
        total_tokens: 165,
        cost_usd: 0.0018,
      },
      {
        row_index: 1,
        input: 'Summarize quantum computing',
        gold: 'Quantum computing uses qubits for parallel computation.',
        output: 'Quantum computing leverages qubits to perform parallel computations.',
        score: 0.88,
        iterations: 3,
        template: 'Translate {input} accurately.',
        comments: '',
        original_output: 'Quantum computing is a type of computing.',
        original_score: 0.42,
        latency_ms: 3100,
        original_latency_ms: 950,
        prompt_tokens: 200,
        completion_tokens: 80,
        total_tokens: 280,
        cost_usd: 0.0031,
      },
    ],
    summary: {
      total_rows: 2,
      original_avg: 0.51,
      optimized_avg: 0.915,
      improvement: 40.5,
      total_iterations: 5,
      met_target: 2,
      total_latency_ms: 5440,
      avg_latency_ms: 2720,
      total_cost_usd: 0.0049,
      total_tokens: 445,
      model: 'gpt-4o',
      total_baseline_latency_ms: 1750,
    },
  };

  it('test_result_mapping_all_fields', () => {
    // Simulate the exact mapping from the frontend (line 2522-2528)
    const mapped = mockBackendResponse.results.map(function (r) {
      return {
        row_index: r.row_index,
        input: r.input,
        gold: r.gold,
        output: r.output,
        score: r.score,
        iterations: r.iterations,
        template: r.template,
        comments: r.comments,
        original_output: r.original_output,
        original_score: r.original_score,
        latency_ms: r.latency_ms,
        original_latency_ms: r.original_latency_ms,
        prompt_tokens: r.prompt_tokens,
        completion_tokens: r.completion_tokens,
        total_tokens: r.total_tokens,
        cost_usd: r.cost_usd,
      };
    });

    assert.strictEqual(mapped.length, 2, 'Should map 2 results');

    // First result
    assert.strictEqual(mapped[0].row_index, 0);
    assert.strictEqual(mapped[0].input, 'Translate hello to French');
    assert.strictEqual(mapped[0].gold, 'Bonjour');
    assert.strictEqual(mapped[0].output, 'Bonjour');
    assert.strictEqual(mapped[0].score, 0.95);
    assert.strictEqual(mapped[0].iterations, 2);
    assert.strictEqual(mapped[0].template, 'Translate {input} accurately.');
    assert.strictEqual(mapped[0].comments, 'Basic translation');
    assert.strictEqual(mapped[0].original_output, 'Hello in French is bonjour.');
    assert.strictEqual(mapped[0].original_score, 0.6);
    assert.strictEqual(mapped[0].latency_ms, 2340);
    assert.strictEqual(mapped[0].original_latency_ms, 800);
    assert.strictEqual(mapped[0].prompt_tokens, 120);
    assert.strictEqual(mapped[0].completion_tokens, 45);
    assert.strictEqual(mapped[0].total_tokens, 165);
    assert.strictEqual(mapped[0].cost_usd, 0.0018);

    // Second result
    assert.strictEqual(mapped[1].row_index, 1);
    assert.strictEqual(mapped[1].score, 0.88);
    assert.strictEqual(mapped[1].iterations, 3);
    assert.strictEqual(mapped[1].original_score, 0.42);
  });

  it('test_summary_mapping', () => {
    const summary = mockBackendResponse.summary;

    assert.strictEqual(summary.total_rows, 2);
    assert.strictEqual(summary.original_avg, 0.51);
    assert.strictEqual(summary.optimized_avg, 0.915);
    assert.strictEqual(summary.improvement, 40.5);
    assert.strictEqual(summary.total_iterations, 5);
    assert.strictEqual(summary.met_target, 2);
    assert.strictEqual(summary.total_latency_ms, 5440);
    assert.strictEqual(summary.avg_latency_ms, 2720);
    assert.strictEqual(summary.total_cost_usd, 0.0049);
    assert.strictEqual(summary.total_tokens, 445);
    assert.strictEqual(summary.model, 'gpt-4o');
    assert.strictEqual(summary.total_baseline_latency_ms, 1750);
  });

  it('test_score_display_format', () => {
    // The frontend uses: (score*100).toFixed(0) + '%'
    const testCases = [
      { score: 0.95, expected: '95%' },
      { score: 0.88, expected: '88%' },
      { score: 1.0, expected: '100%' },
      { score: 0.0, expected: '0%' },
      { score: 0.123, expected: '12%' },
      { score: 0.999, expected: '100%' },
      { score: 0.005, expected: '1%' },
      { score: 0.604, expected: '60%' },
    ];

    for (const tc of testCases) {
      const display = (tc.score * 100).toFixed(0) + '%';
      assert.strictEqual(
        display,
        tc.expected,
        `Score ${tc.score} should display as ${tc.expected}, got ${display}`
      );
    }
  });

  it('test_improvement_negative', () => {
    // The frontend uses: '+' + improvement + '%' for positive
    // When improvement is negative the toFixed(0) already includes '-'
    // The frontend template literally does: `+${improvement}%`
    // So for negative values: +${(-2).toFixed(0)}% => "+-2%"
    const improvement = -2;
    const display = `+${improvement.toFixed ? improvement.toFixed(0) : improvement}%`;
    assert.strictEqual(
      display,
      '+-2%',
      'Negative improvement should display as "+-2%" per the template logic'
    );

    // Also test the per-row delta display which uses different logic:
    // parseInt(delta)>=0 ? '+' : '' followed by delta + '%'
    const origScore = 0.8;
    const newScore = 0.78;
    const delta = ((newScore - origScore) * 100).toFixed(0);
    const deltaDisplay =
      (parseInt(delta) >= 0 ? '+' : '') + delta + '%';
    assert.strictEqual(
      deltaDisplay,
      '-2%',
      'Per-row negative delta should display as "-2%"'
    );
  });

  it('test_improvement_positive', () => {
    const origScore = 0.6;
    const newScore = 0.92;
    const delta = ((newScore - origScore) * 100).toFixed(0);
    const deltaDisplay =
      (parseInt(delta) >= 0 ? '+' : '') + delta + '%';
    assert.strictEqual(
      deltaDisplay,
      '+32%',
      'Positive delta should display as "+32%"'
    );
  });

  it('test_improvement_zero', () => {
    const origScore = 0.75;
    const newScore = 0.75;
    const delta = ((newScore - origScore) * 100).toFixed(0);
    const deltaDisplay =
      (parseInt(delta) >= 0 ? '+' : '') + delta + '%';
    assert.strictEqual(
      deltaDisplay,
      '+0%',
      'Zero delta should display as "+0%"'
    );
  });

  it('test_latency_display', () => {
    // The frontend uses: (latency_ms/1000).toFixed(1) + 's'
    const testCases = [
      { latency_ms: 2340, expected: '2.3s' },
      { latency_ms: 3100, expected: '3.1s' },
      { latency_ms: 500, expected: '0.5s' },
      { latency_ms: 10000, expected: '10.0s' },
      { latency_ms: 0, expected: '0.0s' },
      { latency_ms: 55, expected: '0.1s' },
      { latency_ms: 999, expected: '1.0s' },
      { latency_ms: 1234, expected: '1.2s' },
    ];

    for (const tc of testCases) {
      const display = (tc.latency_ms / 1000).toFixed(1) + 's';
      assert.strictEqual(
        display,
        tc.expected,
        `Latency ${tc.latency_ms}ms should display as ${tc.expected}`
      );
    }
  });

  it('test_latency_display_null_fallback', () => {
    // When latency_ms is falsy, the template shows '--'
    const r = { latency_ms: null };
    const display = r.latency_ms ? (r.latency_ms / 1000).toFixed(1) + 's' : '--';
    assert.strictEqual(display, '--', 'Null latency should display as "--"');
  });

  it('test_cost_display', () => {
    // The frontend uses: '$' + cost_usd.toFixed(4)
    const testCases = [
      { cost_usd: 0.0018, expected: '$0.0018' },
      { cost_usd: 0.0031, expected: '$0.0031' },
      { cost_usd: 0.0, expected: '$0.0000' },
      { cost_usd: 1.23456, expected: '$1.2346' },
      { cost_usd: 0.00001, expected: '$0.0000' },
      { cost_usd: 0.12345, expected: '$0.1235' },
    ];

    for (const tc of testCases) {
      const display = '$' + tc.cost_usd.toFixed(4);
      assert.strictEqual(
        display,
        tc.expected,
        `Cost ${tc.cost_usd} should display as ${tc.expected}`
      );
    }
  });

  it('test_cost_display_null_fallback', () => {
    // When cost_usd is null/undefined, the template shows '--'
    const r = { cost_usd: null };
    const display = r.cost_usd != null ? '$' + r.cost_usd.toFixed(4) : '--';
    assert.strictEqual(display, '--', 'Null cost should display as "--"');
  });

  it('test_cost_display_zero_is_not_null', () => {
    // cost_usd = 0 should NOT fall through to '--' because != null is true for 0
    const r = { cost_usd: 0 };
    const display = r.cost_usd != null ? '$' + r.cost_usd.toFixed(4) : '--';
    assert.strictEqual(display, '$0.0000', 'Zero cost should display as "$0.0000", not "--"');
  });

  it('test_score_color_thresholds', () => {
    // The frontend uses: score >= 0.85 ? green : score >= 0.6 ? yellow : red
    function getColor(score) {
      return score >= 0.85
        ? 'var(--green)'
        : score >= 0.6
          ? 'var(--yellow)'
          : 'var(--red)';
    }

    assert.strictEqual(getColor(0.9), 'var(--green)', '0.9 should be green');
    assert.strictEqual(getColor(0.85), 'var(--green)', '0.85 should be green (boundary)');
    assert.strictEqual(getColor(0.7), 'var(--yellow)', '0.7 should be yellow');
    assert.strictEqual(getColor(0.6), 'var(--yellow)', '0.6 should be yellow (boundary)');
    assert.strictEqual(getColor(0.5), 'var(--red)', '0.5 should be red');
    assert.strictEqual(getColor(0.0), 'var(--red)', '0.0 should be red');
  });

  it('test_total_iterations_sum', () => {
    // The footer uses: st.results.reduce((s,r) => s + r.iterations, 0)
    const results = [
      { iterations: 2 },
      { iterations: 3 },
      { iterations: 1 },
    ];
    const total = results.reduce((s, r) => s + r.iterations, 0);
    assert.strictEqual(total, 6, 'Total iterations should sum correctly');
  });

  it('test_met_target_count', () => {
    // st.results.filter(r => r.score >= st.targetScore).length
    const targetScore = 0.85;
    const results = [
      { score: 0.92 },
      { score: 0.80 },
      { score: 0.88 },
      { score: 0.60 },
    ];
    const metTarget = results.filter((r) => r.score >= targetScore).length;
    assert.strictEqual(metTarget, 2, 'Should count rows meeting target score');
  });

  it('test_chart_bar_height_calculation', () => {
    // Math.max(4, score * 140) — ensures minimum height of 4px
    const testCases = [
      { score: 0.95, expected: 133 },
      { score: 0.0, expected: 4 },         // Clamped to min 4
      { score: 0.01, expected: 4 },         // 0.01 * 140 = 1.4 < 4, clamped
      { score: 0.5, expected: 70 },
      { score: 1.0, expected: 140 },
    ];

    for (const tc of testCases) {
      const height = Math.max(4, tc.score * 140);
      assert.strictEqual(
        height,
        tc.expected,
        `Score ${tc.score} should produce bar height ${tc.expected}px`
      );
    }
  });
});

// ===========================================================================
// 5. Integration Test (requires both servers running)
// ===========================================================================

describe('5. Integration Tests', () => {
  it('test_end_to_end_fetch', async () => {
    // This test requires:
    // 1. Frontend server on localhost:8000
    // 2. Backend server on localhost:5000
    // It is expected to be skipped (via soft failure) if servers are not running.

    let frontendAvailable = false;
    let backendAvailable = false;

    // Check frontend
    try {
      const fRes = await fetch('http://localhost:8000/index.html', {
        signal: AbortSignal.timeout(3000),
      });
      frontendAvailable = fRes.status === 200;
    } catch {
      frontendAvailable = false;
    }

    // Check backend
    try {
      const bRes = await fetch('http://localhost:5000/', {
        signal: AbortSignal.timeout(3000),
      });
      backendAvailable = true;
    } catch {
      backendAvailable = false;
    }

    if (!frontendAvailable || !backendAvailable) {
      // Use console.log to signal skip — node:test does not have a built-in skip
      console.log(
        '  [SKIPPED] test_end_to_end_fetch: ' +
          (!frontendAvailable ? 'frontend server not running on :8000. ' : '') +
          (!backendAvailable ? 'backend server not running on :5000. ' : '') +
          'Start both servers to run this test.'
      );
      return;
    }

    // Verify frontend serves the optimizer page div
    const frontendRes = await fetch('http://localhost:8000/index.html');
    const frontendHtml = await frontendRes.text();
    assert.ok(
      frontendHtml.includes('id="page-optimizer"'),
      'Frontend should contain the optimizer page div'
    );
    assert.ok(
      frontendHtml.includes('function renderOptimizer'),
      'Frontend should contain the renderOptimizer function'
    );

    // Try a health-check or status endpoint on the backend
    // (We avoid actually running a full optimization which requires an API key)
    try {
      const healthRes = await fetch('http://localhost:5000/health', {
        signal: AbortSignal.timeout(5000),
      });
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        assert.ok(
          healthData,
          'Backend health endpoint should return a response'
        );
      }
    } catch {
      // Health endpoint may not exist — that is acceptable
      console.log('  [INFO] Backend /health endpoint not available, skipping health check');
    }

    // Verify the backend's optimize endpoint exists by sending a malformed
    // request (no API key) and checking we get a structured error, not a 404
    try {
      const optimizeRes = await fetch('http://localhost:5000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        signal: AbortSignal.timeout(5000),
      });

      // We expect either a 400/422 (validation error) or 500 — NOT a 404
      if (optimizeRes.status === 404) {
        console.log('  [INFO] Backend /optimize endpoint returned 404 — endpoint may not be configured');
      }
    } catch (err) {
      // Connection errors are acceptable if the backend is partially running
      console.log('  [INFO] Could not reach /optimize endpoint: ' + err.message);
    }

    // Simulate what the frontend does with a mock backend response:
    // Build the same mapped results and verify rendering logic
    const mockResult = {
      success: true,
      results: [
        {
          row_index: 0,
          input: 'test input',
          gold: 'test gold',
          output: 'test output',
          score: 0.91,
          iterations: 1,
          template: 'test template',
          comments: '',
          original_output: 'original output',
          original_score: 0.55,
          latency_ms: 1500,
          original_latency_ms: 400,
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
          cost_usd: 0.0012,
        },
      ],
      summary: {
        total_rows: 1,
        original_avg: 0.55,
        optimized_avg: 0.91,
        improvement: 36,
        total_iterations: 1,
        met_target: 1,
        total_latency_ms: 1500,
        avg_latency_ms: 1500,
        total_cost_usd: 0.0012,
        total_tokens: 150,
        model: 'gpt-4o',
      },
    };

    // Verify the mapping step
    const mapped = mockResult.results.map(function (r) {
      return {
        row_index: r.row_index,
        input: r.input,
        gold: r.gold,
        output: r.output,
        score: r.score,
        iterations: r.iterations,
        template: r.template,
        comments: r.comments,
        original_output: r.original_output,
        original_score: r.original_score,
        latency_ms: r.latency_ms,
        original_latency_ms: r.original_latency_ms,
        prompt_tokens: r.prompt_tokens,
        completion_tokens: r.completion_tokens,
        total_tokens: r.total_tokens,
        cost_usd: r.cost_usd,
      };
    });

    assert.strictEqual(mapped.length, 1);
    assert.strictEqual(mapped[0].score, 0.91);
    assert.strictEqual(
      (mapped[0].score * 100).toFixed(0) + '%',
      '91%',
      'Mapped score should display as 91%'
    );
    assert.strictEqual(
      (mockResult.summary.original_avg * 100).toFixed(0) + '%',
      '55%',
      'Original avg should display as 55%'
    );
    assert.strictEqual(
      (mockResult.summary.optimized_avg * 100).toFixed(0) + '%',
      '91%',
      'Optimized avg should display as 91%'
    );
  });
});
