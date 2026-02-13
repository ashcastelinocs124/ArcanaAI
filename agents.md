# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an open-source LLM observability platform that performs forensic analysis on multi-agent execution traces. The system reconstructs agent DAGs (Directed Acyclic Graphs), identifies communication issues between agents, and monitors task progress using LLM-based verification.

**Core Purpose:** Given agent execution logs with parent/child relationships, build DAGs to analyze:
- Agent communication quality (input/output similarity across edges)
- Task progress monitoring (whether agents are executing the user's goal correctly)
- Multi-agent architecture patterns (linear, swarm, fan-out, diamond)

## Commands

### Running the Main Pipeline
```bash
# Run with default trace and user goal
python semantic_pipeline.py

# Run with custom user goal and trace ID
python semantic_pipeline.py "Book a flight from London to Tokyo for 2 adults." "tr-a7f2-9b3c-4e1d-8f6a"
```

### Testing
```bash
# Run all tests
python -m unittest test_agent_dag

# Run specific test class
python -m unittest test_agent_dag.TestAgentDAG

# Run specific test
python -m unittest test_agent_dag.TestAgentDAG.test_from_agent_logs_returns_dict_of_dags
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key in .env (required for TaskProgressMonitor LLM calls)
# OPENAI_API_KEY=your-key-here
```

## Frontend Reference

When modifying `frontend/index.html`, read `ai/claude/arcana-frontend.md` first. It documents the single-file architecture, 5-point navigation system, critical bug patterns (DOM re-render file loss, Excel column mismatches), rendering patterns, and CSS design system.

## Context Summaries

Summaries are stored in `context/` as skill-structured markdown files.

When you need prior context:
- Select the summary file **by matching the title** in its frontmatter to the user's query.
- Use the chosen summary file for relevant background before proceeding.

## Architecture

### Data Flow
1. **Extract** → `extract_agent_logs()` flattens nested journey structure from `keywords_ai_agent_output_samples.json`
2. **Build DAG** → `AgentDAG.from_agent_logs()` constructs parent/child relationships per trace_id
3. **Analyze** → `EmbeddingMatchAnalyzer` or `TaskProgressMonitor` process the DAG

### Core Classes

#### AgentDAG (`semantic_pipeline.py:46-147`)
Represents the execution graph for one journey (trace_id). Builds parent/child relationships from agent logs using `parent_id` or `parent_ids` fields.

**Key Concepts:**
- **One DAG per trace_id**: All agents in a journey share the same trace_id
- **Multi-architecture support**: Handles linear chains, fan-out (1→N), fan-in (N→1), and diamond patterns
- **Node = agent execution**: Each node contains full agent data (input, output, tool_calls, metadata)
- **Edge = (parent_id, child_id)**: Represents control flow between agents

**Key Methods:**
- `from_agent_logs(logs)` → dict[trace_id, AgentDAG]: Factory to build all DAGs from flat log list
- `get_parents(agent_id)` / `get_children(agent_id)`: Navigation
- `roots()` / `leaves()`: Find entry/exit points
- `get_node(agent_id)`: Retrieve full agent data

#### EmbeddingMatchAnalyzer (`semantic_pipeline.py:177-303`)
Detects communication issues by measuring input/output similarity across edges.

**How it works:**
- For each edge (parent → child): compute `similarity = cosine_sim(child.input, child.output)`
- Mark edge RED if similarity < threshold (default 0.8)
- Compute running average and standard deviation of similarity scores

**Use case:** If child.input (what parent said) doesn't match child.output (what child produced), communication may have broken down.

**Key Methods:**
- `analyze()`: Process all edges, return list of {edge, similarity, diff, status}
- `print_red_agents()`: Display full context for agents with similarity < threshold
- `avg` / `stdev`: Statistical properties of similarity distribution

#### TaskProgressMonitor (`semantic_pipeline.py:320-419`)
Validates whether agents are executing the user's intended goal using periodic LLM checkpoints.

**How it works:**
- Takes user_goal (e.g., "Book a flight from London to Tokyo")
- Every N steps (default 5), summarizes agent actions so far
- Calls LLM with: "User goal: X. Agent progress: Y. Are they on track?"
- Returns checkpoints with LLM verdicts

**Use case:** Detect when agent swarm starts drifting from the original task.

### Data Schema

Agent logs from `keywords_ai_agent_output_samples.json` contain:
- `trace_id`: Journey identifier (all agents in same execution share this)
- `journey_name`: Human-readable journey description
- `agent_id`: Unique per agent execution
- `parent_id` or `parent_ids`: Parent agent(s) that triggered this agent
- `input`: Text input to this agent (from user or parent's output)
- `output`: This agent's completion message
- `core_payload`: Contains tool_calls, reasoning_blocks
- `telemetry`: TTFT, latency, token counts
- `metadata`: customer_plan, environment, session_id

### Multi-Agent Architecture Patterns

The DAG supports multiple execution patterns:

**Linear**: A→B→C (single parent per agent)
```
agent-1 → agent-2 → agent-3
```

**Fan-out**: A→(B,C) (one parent, multiple children)
```
agent-1 ──┬→ agent-2
          └→ agent-3
```

**Fan-in**: (A,B)→C (multiple parents, one child)
```
agent-2 ──┬→ agent-4
agent-3 ──┘
```

**Diamond**: Combines fan-out + fan-in
```
agent-1 ──┬→ agent-2 ──┬→ agent-4
          └→ agent-3 ──┘
```

Use `dag.get_parents()` and `dag.get_children()` to navigate these patterns.

## Development Notes

### Adding New Analyzers
Create a new class that takes `AgentDAG` in `__init__`. Follow the pattern:
```python
class MyAnalyzer:
    def __init__(self, dag: AgentDAG):
        self.dag = dag

    def analyze(self) -> list[dict]:
        # Process dag.get_edges() or dag.get_nodes()
        # Return structured results
        pass
```

### Working with Embeddings
The current `_text_similarity()` uses simple cosine similarity on term frequencies. To use real embeddings:
1. Replace `_text_similarity()` with a function that calls an embedding API
2. Pass it to `EmbeddingMatchAnalyzer(dag, similarity_fn=your_fn)`

### LLM Configuration
`TaskProgressMonitor` uses LiteLLM for model calls. Supports:
- OpenAI models (requires `OPENAI_API_KEY` in `.env`)
- Local models via Ollama: `model='ollama/llama2'`
- Any LiteLLM-supported provider

### Testing Multi-Agent Patterns
Test data in `keywords_ai_agent_output_samples.json` includes:
- `tr-a7f2-9b3c-4e1d-8f6a`: Trip Booking (fan-out + fan-in, 6 agents)
- `tr-b8e3-0c4d-5f2e-9a7b`: Refund (fan-out from root)
- `tr-c9f4-1d5e-6a3f-0b8c`: Quote (diamond pattern)

Use these trace IDs to test different DAG topologies.

## Pre-Task Checklist (ALWAYS READ BEFORE WORKING)

**Before solving any user query, review these learnings. They encode hard-won patterns and prevent repeated mistakes.**

### Deployment Verification
- **Always verify changes are live** after modifying frontend files. The dev server at `localhost:8000` serves files directly from `frontend/`. After every edit, `curl` the served file and grep for a unique string from your change.
- The server is a simple Python HTTP server (`python3 -m http.server 8000` in `frontend/`). No build step — file edits are immediately live on refresh.
- **After every major change, verify BOTH frontend and backend are running:**
  - Frontend: `curl -s http://localhost:8000/index.html | head -1` (should return HTML)
  - Backend: `curl -s http://localhost:5000/health` (should return OK)
  - If either is down, restart: frontend with `cd frontend && python3 -m http.server 8000 &`, backend with `python app.py &` (or however the backend starts)
- **Data must stay in sync**: When adding or modifying agent data, update BOTH `test/keywords_ai_agent_output_samples.json` (backend) AND `frontend/data.json` (frontend). Never update just one — verify both files have identical journeys and agent IDs after changes.

### Frontend Architecture Gotchas
- **Single-file frontend**: `frontend/index.html` contains ALL HTML, CSS, and JS (~2500+ lines). There is no build system, no bundler, no framework.
- **Syntax validation**: `node --check` cannot validate `.html` files. Extract JS with: `python3 -c "import re; ..."` into a `.js` file, then run `node --check` on that. Always validate after edits.
- **DOM re-render destroys state**: File inputs (`<input type="file">`) lose their reference when `innerHTML` is reassigned. That's why `optimizerState.file` stores the File object separately — always follow this pattern.
- **Function scope**: Functions called from `onclick` in rendered HTML must be assigned to `window` (e.g., `window.renderOptimizer = renderOptimizer`).

### DAG Viewer Design Decisions
- **Nodes turn red, NOT edges**: When an agent has high input/output drift (similarity < 0.8), the node itself turns red — not the connecting lines. This was an explicit user preference. The `redNodeMap` in `renderDAGWithEdgeColors()` tracks which child nodes are flagged.
- **Always explain WHY**: When flagging drift, show the user the reason. The DAG viewer has a "Drift Analysis" section that shows: similarity %, a human-readable explanation (4 severity tiers), and side-by-side input vs output comparison.
- **Agent panel shows drift too**: `showAgentPanel()` checks if the clicked agent is in a red edge and displays a "High Drift Detected" banner with the explanation.
- **Hallucination Detection (LLM verification)**: `HallucinationDetector` in `semantic_pipeline.py` runs a second-pass LLM check on RED-flagged edges. Compares agent output to the user's original goal via `gpt-4o-mini`. Three outcomes: `cleared` (false positive → turns green), `verified_red` (confirmed hallucination → stays red), `unverified` (LLM error → stays red). Only runs when `user_goal` is provided. Frontend shows "LLM OK" badge on cleared nodes, "Cleared by LLM" / "Hallucinating" badges on drift cards, and an "LLM Check" column in the edge table.

### Data Schema (Two Formats)
- **`frontend/data.json`** (journey-grouped schema): Used by the frontend. Has `journeys[]` array, each with `samples[]`. Fields: `agent_turn`, `agent_name`, `core_payload.completion_message`, `telemetry.ttft/latency/tokens`, `model_parameters.model`, `evaluation_signals`, `agent_id`, `parent_id`/`parent_ids`, `input`, `output`.
- **`data.json`** (flat record schema): Simpler format with `records[]` array. Fields: `trace_id`, `agent_id`, `parent_id`, `agent_name`, `input`, `output`, `telemetry.latency_ms/ttft_ms/tokens_in/tokens_out`, `metadata.model`.
- **When merging flat records into frontend data**: Group by `trace_id` into journeys, convert `latency_ms` → seconds, `tokens_in/out` → `prompt_tokens/completion_tokens`, create root agents for referenced-but-missing parent IDs.

### Optimizer State
- **`selectedAgent`**: The optimizer supports per-agent filtering. When an Excel file is uploaded with an agent column, a dropdown appears. Only that agent's rows are processed.
- **Reset on new upload**: `selectedAgent` auto-selects the first agent on file upload, or `null` if no agent column exists.
- **Backend future-proofing**: `agent_filter` is appended to FormData for the backend path even though the backend doesn't filter yet.

### Landing Page (`frontend/landing.html`)
- Separate from the main app (`frontend/index.html`). Dark theme with copper accent (`#c8956c`).
- **Animations added**: Mouse parallax on hero cards, 3D card tilt on hover, magnetic buttons, animated counters, SVG edge drawing, code typewriter effect, scroll progress bar, cursor glow, mini-bar growth, staggered node/table reveals, step connection line + ripple.
- **All animations respect `prefers-reduced-motion`** — disabled when the OS setting is on.
- **Mobile handling**: Parallax and cursor effects are disabled below 768px.

### Testing Patterns
- **Frontend JS tests**: Use Node.js with indirect eval `(0, eval)(code)` to load functions into global scope. Balance-brace parsing extracts individual functions from the HTML.
- **Test files**: Stored in `/tmp/test_dag*.js`. Run with `node /tmp/test_dag3.js`.

### Common Mistakes to Avoid
1. **Don't color edges red** — color the nodes instead (user preference)
2. **Don't skip deployment verification** — always curl localhost:8000 after changes
3. **Don't use `node --check` on .html** — extract JS first
4. **Don't forget `window.functionName`** — inline onclick handlers need global scope
5. **Don't assume data format** — check whether you're working with journey-grouped or flat-record schema
6. **Don't just flag problems — explain them** — users need to understand WHY something is wrong, not just that it is
7. **Don't send all journeys to backend without `trace_id`** — cascade evaluation makes ~7 LLM calls per trace; sending 15 traces = ~100 calls = timeout. Always filter with `trace_id` when evaluating a specific trace. *(Feb 9 2026)*
8. **Don't run per-agent LLM evaluations sequentially** — if agent evaluations are independent (no shared state), parallelize with `ThreadPoolExecutor`. Sequential: N × latency. Parallel: 1 × latency. *(Feb 9 2026)*
9. **Workflow system prompts don't have `{input}`** — the optimizer requires `{input}` placeholder to inject test data. Workflow agents use system-prompt architecture (fixed instruction + user message), not template-based prompts. `createPromptFromCascade()` must append `\n\nUser request: {input}` if missing. *(Feb 9 2026)*
10. **Optimizer `{input}` recovery** — LLMs sometimes drop the `{input}` placeholder from improved prompts. Instead of `break`ing the loop (giving up), append `\n\n{input}` to the LLM's output to salvage the improvement. *(Feb 9 2026)*
11. **`isWorkflowSource` short-circuits the optimizer** — `runOptimizer()` has two paths: workflow re-execution (for testing prompts by re-running the workflow) and actual optimization (backend Celery / client-side). Cascade prompts have `source: 'workflow'`, so they hit the workflow path and the optimizer never runs. Fix: only take the workflow path when there's no uploaded Excel file. *(Feb 9 2026)*
12. **Agent names in `data.json` don't match workflow topology** — Static data uses `EarningsAnalystAgent` but topology has `EarningsAnalyst` (no "Agent" suffix). This breaks both topology lookup (no system prompt found) and prompt overrides (wrong key). Fix: fuzzy match in `createPromptFromCascade` + store `topoAgentName` separately for override keys. *(Feb 9 2026)*
13. **Celery worker doesn't auto-reload** — Unlike Django's `runserver`, Celery workers load code once at startup. After changing `tasks.py` (or any module it imports), you **must restart the worker** or it runs stale code. Symptoms: `TypeError: unexpected keyword argument` or silently wrong behavior. Fix: restart worker, or use `celery -A arcana worker --loglevel=info --autoreload` (requires `watchdog`). *(Feb 11 2026)*

### Lessons Learned (Timestamped)

#### Feb 9, 2026
- **`loadPipelineFromBackend()` must accept `traceId`**: Without it, the backend evaluates every trace in the dataset. For cascade mode with LLM calls, this causes silent timeouts. Always scope API calls to the trace the user is actually looking at.
- **Merge results, don't overwrite**: When evaluating one trace at a time, use `Object.assign(pipelineState.results, newResults)` instead of replacing. This lets the user switch between traces without losing prior evaluations.
- **`_call_llm()` is thread-safe**: LiteLLM's `completion()` is stateless — safe to call from multiple threads. The `SpecialistEvaluator.analyze()` loop was a free parallelization win (same pattern as `workflow_engine.py`'s `ThreadPoolExecutor`).
- **Flask `debug=False` is required for SSE**: Werkzeug's debug middleware buffers entire streaming responses. This was fixed in a prior session but worth remembering — any Flask SSE endpoint will appear to hang if `debug=True`.
- **Profiling LLM-heavy endpoints**: When an endpoint is slow, count total LLM calls × model latency. `gpt-4o` = 3-8s/call, `gpt-4o-mini` = 1-3s/call. Multiply by number of sequential calls to get expected latency. This gives you the optimization target immediately.

### Self-Improvement Rule
Whenever you learn something interesting — whether from a mistake, a debugging session, a surprising behavior, or a new pattern — **add it to this CLAUDE.md file** under the appropriate section (e.g., "Common Mistakes to Avoid", "Frontend Architecture Gotchas", "DAG Viewer Design Decisions", etc.). If no existing section fits, create a new one. This ensures hard-won knowledge is preserved across sessions and never re-learned the painful way.

### Context Preservation on Auto-Compact
When context is auto-compacted (conversation history is compressed), **immediately write a context summary** to `context/` before continuing. Follow this format:
1. Create a file: `context/conversation-summary-YYYY-MM-DD-<topic>.md`
2. Use the same frontmatter format as existing summaries (name, title, description, tags)
3. Include sections: **Overview**, **Key Activities**, **Technical Changes** (files created/modified with line refs), **Architecture Decisions**, **Deployment Notes**
4. Capture everything needed to resume work without the full conversation: what was built, why, what files changed, what's running, and any gotchas encountered
5. This is **mandatory** — do not skip it even if the compaction seems minor
6. **Housekeeping**: After writing a new summary, check the total number of files in `context/`. If there are **more than 5**, merge the 2 oldest files into a single combined summary (preserve all key details, deduplicate overlapping content) and delete the 2 originals
