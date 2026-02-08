# Arcana Quick Start Guide

Get up and running with Arcana in 5 minutes.

## Prerequisites

- Python 3.8+
- OpenAI API key (for prompt optimizer and evaluations)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Step 1: Install Dependencies

```bash
cd /Users/ash/Desktop/interview-prep
pip install -r requirements.txt
```

This installs:
- `litellm` - LLM integration library
- `pandas` - Data processing
- `openpyxl` - Excel file handling
- `flask` - Backend API server
- `flask-cors` - CORS support for frontend
- `python-dotenv` - Environment variable management

## Step 2: Set API Key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or export it directly:

```bash
export OPENAI_API_KEY=your-api-key-here
```

## Step 3: Start Arcana

### Option 1: Quick Start (Recommended)

```bash
./start.sh
```

This will:
- Start the backend API server on `http://localhost:5000`
- Start the frontend server on `http://localhost:8000`
- Automatically open your browser to the dashboard

To stop:

```bash
./stop.sh
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
python3 backend/api.py
```

You should see:
```
Starting Arcana API server on http://localhost:5000
 * Running on http://127.0.0.1:5000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## What You Can Do Now

### 1. **Browse Agent Traces**

- Click **"Traces"** in the sidebar
- See all 9 multi-agent execution journeys
- Click any trace to see detailed execution timeline with DAG visualization

### 2. **Analyze Communication Issues**

- Click **"Semantic Pipeline"** under Analysis
- See edge similarity analysis across agent DAGs
- Red edges indicate communication drift (similarity < 0.8)
- Backend automatically analyzes all traces on first load

### 3. **Optimize Prompts**

- Click **"Prompt Optimizer"** under Analysis
- Upload an Excel/CSV file with columns: `input`, `gold`, `comments` (optional)
  - Sample file available at: `test/data/prompt_optimizer_sample.xlsx`
- Enter your prompt template with `{input}` placeholder
- Click **Run** to optimize
- View side-by-side comparison of original vs optimized results

### 4. **Run Workflows** (Placeholder)

- Click **"DAG Viewer"** under Workbench
- Click **"▶ Run Workflow"** button
- Enter a goal and select workflow type
- Click **Execute**
- (Full implementation pending)

## Testing the Backend APIs

### Test Pipeline Analysis

```bash
curl -X POST http://localhost:5000/api/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "traces": [...],  # Load from keywords_ai_agent_output_samples.json
    "threshold": 0.8
  }'
```

### Test Prompt Optimizer

```bash
curl -X POST http://localhost:5000/api/optimizer/run \
  -F "file=@test/data/prompt_optimizer_sample.xlsx" \
  -F "prompt_template=You are helpful. {input}" \
  -F "target_score=0.85" \
  -F "max_iters=3"
```

## Troubleshooting

### Backend not connecting

**Problem**: Frontend shows "Backend API Error"

**Solution**:
1. Check backend is running: `curl http://localhost:5000/health`
2. Should return: `{"status":"ok","service":"arcana-api","version":"1.0.0"}`
3. If not running, restart: `python backend/api.py`

### LLM Calls Failing

**Problem**: Prompt optimizer returns errors

**Solution**:
1. Check API key is set: `echo $OPENAI_API_KEY`
2. Check backend logs for LiteLLM errors
3. Try local model: Change `eval_model` to `ollama/llama2` (requires Ollama)

### CORS Errors

**Problem**: Browser console shows CORS errors

**Solution**:
1. Make sure Flask-CORS is installed: `pip install flask-cors`
2. Restart backend server
3. Clear browser cache and reload

### File Upload Not Working

**Problem**: Excel upload fails in Prompt Optimizer

**Solution**:
1. Check file format: Must be `.xlsx`, `.xls`, or `.csv`
2. Check file has required columns: `input`, `gold`
3. Check file size: Max 16MB
4. Check backend logs for parsing errors

## Next Steps

1. **Add Your Own Data**
   - Replace `test/keywords_ai_agent_output_samples.json` with your agent traces
   - Follow the schema documented in `CLAUDE.md`

2. **Create Custom Evaluations**
   - Use the `/api/evaluations/run` endpoint
   - Implement custom scoring logic in `backend/api.py`

3. **Implement Workflow Execution**
   - Build agent orchestration logic
   - Connect to `/api/workflow/run` endpoint
   - Stream results back to frontend

4. **Customize the Frontend**
   - Edit `frontend/index.html`
   - All CSS/JS is inline for easy modification
   - Color scheme defined in `:root` CSS variables

## File Structure

```
interview-prep/
├── frontend/
│   └── index.html              # Single-file dashboard
├── backend/
│   └── api.py                  # Flask API server
├── test/
│   ├── keywords_ai_agent_output_samples.json  # Sample data (9 journeys)
│   └── data/
│       └── prompt_optimizer_sample.xlsx       # Sample Excel
├── .env                        # API keys (create this)
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation
└── QUICKSTART.md              # This file
```

## Support

For issues or questions:
- Check `README.md` for detailed documentation
- Review `CLAUDE.md` for architecture details
- See example traces in `test/keywords_ai_agent_output_samples.json`
