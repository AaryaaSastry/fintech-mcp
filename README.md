# Fintech AI Analytics Platform

A modular fintech analytics system using Model Context Protocol (MCP) architecture.

## Architecture
- **Data Layer**: Raw and processed transaction data.
- **Processing Layer**: Feature engineering and data cleaning.
- **Models**: Fraud detection and spending forecasting.
- **MCP Server**: Analytics engine hosting specialized tools.
- **Backend API**: FastAPI gateway for querying tools and data.
- **Frontend**: Dashboard for visualization and risk assessment.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing: `python processing/preprocess.py` (if script exists)

### One-Command Execution
To run both the Frontend and Backend simultaneously:
```powershell
python run_all.py
```
This will start the API on port 8000 and the Dashboard on port 3000, then automatically open your browser.

### Manual Startup
1. Start MCP Server: `python mcp_server/server.py`
2. Start Backend: `python backend/main.py`
3. Start Frontend: Open `frontend/index.html` in your browser.
