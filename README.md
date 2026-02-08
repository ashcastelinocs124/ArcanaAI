# Arcana - LLM Observability Platform

A professional observability platform for multi-agent LLM systems with forensic analysis capabilities.

## Features

### 1. **Dashboard Overview**
- Real-time metrics for traces, agents, latency, and edge issues
- Latency timeline visualization
- Health ring showing pass/fail rates
- Model distribution charts
- Pipeline health per-trace analysis
- Guardrail events monitoring

### 2. **Trace Monitoring**
- Browse all multi-agent execution traces
- Filter by pass/fail status
- Detailed trace views with agent execution timeline
- Interactive DAG visualization showing agent relationships

### 3. **Semantic Pipeline Analysis**
- Detects communication issues between agents using edge similarity
- Computes cosine similarity between child input and parent output
- Highlights RED edges (similarity < 0.8) indicating potential issues
- Visual DAG with color-coded edges (green = pass, red = issue)
- Edge analysis table with similarity percentages

### 4. **Prompt Optimizer**
- Upload Excel/CSV files with test cases (input, gold, comments columns)
- Iteratively optimizes prompt templates using LLM feedback
- Side-by-side comparison: original vs optimized results
- Real-time similarity scoring
- Dashboard showing improvement metrics
- Requires backend API with OpenAI API key

### 5. **DAG Viewer**
- Interactive execution graphs for each trace
- Supports multiple agent patterns:
  - Linear chains (A→B→C)
  - Fan-out (A→B,C)
  - Fan-in (A,B→C)
  - Diamond patterns (combination)
- Run Workflow feature (placeholder - requires backend implementation)

### 6. **Metrics & Logs**
- Latency distribution charts
- Token usage by model
- Agent count by model
- Evaluation score tracking
- Execution logs with timestamps

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## Running the Application

### Quick Start

```bash
./start.sh
```

This starts both servers and opens the dashboard in your browser:
- **Frontend**: http://localhost:8000
- **Backend API**: http://localhost:5000

To stop:

```bash
./stop.sh
```

To check status:

```bash
./status.sh
```

### Manual Start

If you prefer to start servers manually:

**Terminal 1 - Backend API:**
```bash
python3 backend/api.py
# Runs on http://localhost:5000
```

**Terminal 2 - Frontend Server:**
```bash
cd frontend
python3 -m http.server 8000
# Runs on http://localhost:8000
```

Then open http://localhost:8000 in your browser.

### Important Notes

- **Both frontend and backend run on localhost**
- Frontend serves from port 8000 (via Python HTTP server)
- Backend API serves from port 5000 (via Flask)
- Frontend makes CORS requests to backend API
- All API endpoints are at `http://localhost:5000/api/*`

## Using the Prompt Optimizer

### Prepare Your Data

Create an Excel (.xlsx) or CSV file with these columns:
- `input`: The input text for your prompt
- `gold`: The expected/desired output
- `comments`: (Optional) Notes about this test case

Example:

| input | gold | comments |
|-------|------|----------|
| Hello | Hi there! | Greeting test |
| What is 2+2? | 4 | Math question |

### Run Optimization

1. Navigate to **Analysis → Prompt Optimizer** in the sidebar
2. Enter your prompt template with `{input}` placeholder:
   ```
   You are a helpful assistant. Answer concisely. {input}
   ```
3. Set target score (0-1, default 0.85) and max iterations (1-10, default 3)
4. Upload your Excel/CSV file
5. Click **Run**
6. View side-by-side comparison of original vs optimized results

## Running the Semantic Pipeline (Python)

Analyze agent execution traces from the command line:

```bash
# Run with default trace and goal
python semantic_pipeline.py

# Run with custom goal and trace ID
python semantic_pipeline.py "Book a flight from London to Tokyo" "tr-a7f2-9b3c-4e1d-8f6a"
```

## API Endpoints

### POST /api/pipeline/analyze

Run semantic pipeline analysis on agent traces.

**JSON Body:**
```json
{
  "traces": [...],  // Array of journey objects (keywords_ai format)
  "trace_id": "tr-abc...",  // Optional: analyze specific trace
  "threshold": 0.8,  // Optional: similarity threshold (default 0.8)
  "user_goal": "Book a flight..."  // Optional: for task progress monitoring
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "tr-abc...": {
      "trace_id": "tr-abc...",
      "edges": [
        {
          "parent_id": "agent-1",
          "child_id": "agent-2",
          "similarity": 0.95,
          "status": "green",
          "parent_output": "...",
          "child_input": "..."
        }
      ],
      "avg_similarity": 0.92,
      "stdev_similarity": 0.05,
      "red_edges": 0,
      "total_edges": 5,
      "task_progress": [  // Only if user_goal provided
        {
          "step": 0,
          "summary": "...",
          "verdict": "on_track"
        }
      ]
    }
  }
}
```

### POST /api/evaluations/run

Run custom evaluations on agent outputs (placeholder implementation).

**JSON Body:**
```json
{
  "traces": [...],  // Array of agent samples
  "eval_type": "accuracy",  // Type of evaluation
  "eval_prompt": "...",  // Custom evaluation prompt template
  "model": "gpt-4o-mini"  // Optional: LLM model for evaluation
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "agent_id": "agent-1",
      "eval_type": "accuracy",
      "score": 0.85,
      "output": "..."
    }
  ],
  "summary": {
    "total_evaluated": 10,
    "avg_score": 0.85,
    "eval_type": "accuracy"
  }
}
```

### POST /api/optimizer/run

Run prompt optimization with uploaded Excel file.

**Form Data:**
- `file`: Excel/CSV file
- `prompt_template`: Prompt template with `{input}` placeholder
- `target_score`: Target similarity score (0-1)
- `max_iters`: Max iterations per row
- `input_col`: Column name/pattern for input (default: "input")
- `gold_col`: Column name/pattern for gold output (default: "gold")
- `comments_col`: Optional column for comments
- `eval_model`: LLM model for evaluation (default: "gpt-4o-mini")
- `optimizer_model`: LLM model for optimization (default: "gpt-4o-mini")

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "row_index": 0,
      "input": "Hello",
      "gold": "Hi there!",
      "output": "Hi there!",
      "score": 1.0,
      "iterations": 2,
      "template": "Optimized prompt...",
      "comments": null,
      "original_output": "Hello to you",
      "original_score": 0.6
    }
  ],
  "summary": {
    "total_rows": 1,
    "original_avg": 0.6,
    "optimized_avg": 1.0,
    "improvement": 40.0,
    "total_iterations": 2,
    "met_target": 1
  }
}
```

### POST /api/workflow/run

Execute multi-agent workflow (not yet implemented).

**JSON Body:**
```json
{
  "goal": "Book a flight from London to Tokyo",
  "workflow_type": "trip_booking",
  "config": {}
}
```

## Project Structure

```
interview-prep/
├── frontend/
│   └── index.html           # Single-file dashboard (HTML+CSS+JS)
├── backend/
│   ├── api.py              # Flask API server
│   └── pipeline/
│       └── semantic_pipeline.py
├── ai/
│   └── prompt_optimizer.py # Prompt optimization logic
├── test/
│   ├── keywords_ai_agent_output_samples.json  # Sample data
│   ├── data/
│   │   └── prompt_optimizer_sample.xlsx
│   └── test_*.py           # Unit tests
├── semantic_pipeline.py    # CLI for semantic analysis
├── requirements.txt
└── README.md
```

## Technologies

- **Frontend**: Vanilla JavaScript, single HTML file
- **Backend**: Flask with CORS support
- **LLM Integration**: LiteLLM (supports OpenAI, Anthropic, local models)
- **Data Processing**: pandas, openpyxl
- **Styling**: Custom CSS with dark theme, copper accents
- **Fonts**: Instrument Serif (display), DM Sans (body)

## Design System

### Colors
- Background: `#08080c`
- Cards: `#111118`
- Accent: Copper `#c8956c`
- Success: Green `#4ade80`
- Warning: Yellow `#fbbf24`
- Error: Red `#f87171`
- Info: Blue `#60a5fa`

### Typography
- Display: Instrument Serif
- Body: DM Sans
- Code: DM Mono

## Testing

Run Python tests:

```bash
# Run all tests
python -m unittest discover test

# Run specific test file
python -m unittest test_agent_dag
python -m unittest test.test_prompt_optimizer
```

## Architecture

### Multi-Agent DAG
- Each `trace_id` represents one journey with multiple agents
- Agents have `parent_id` or `parent_ids` fields creating DAG structure
- Supports complex patterns: linear, fan-out, fan-in, diamond

### Edge Similarity Analysis
- For each edge (parent → child), compute similarity between:
  - Parent output (what parent said)
  - Child input (what child received)
- Threshold: 0.8 (configurable)
- RED edges indicate communication breakdown

### Prompt Optimization
1. Load Excel with input/gold pairs
2. Run baseline: evaluate original prompt on all inputs
3. Optimization loop per row:
   - Render prompt with input
   - Call LLM, get output
   - Compute similarity to gold
   - If below target, ask optimizer LLM for better prompt
   - Use improved prompt for next iteration
4. Return side-by-side comparison

## Future Enhancements

- [ ] Implement full multi-agent workflow execution
- [ ] Real-time streaming of agent execution
- [ ] Custom agent architecture builder
- [ ] Export reports as PDF
- [ ] Integration with more LLM providers
- [ ] User authentication and project management
- [ ] WebSocket support for live updates
- [ ] Custom evaluation metrics beyond similarity

## License

MIT
