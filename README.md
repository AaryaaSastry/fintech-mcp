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
2. Run preprocessing: `python processing/preprocess.py`
3. Start MCP Server: `python mcp_server/server.py`
4. Start Backend: `python backend/main.py`
5. Run Frontend: (Instructions to follow)
