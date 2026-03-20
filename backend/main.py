from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .mcp_client import start_mcp_client, stop_mcp_client
    from .query_router import router as analytics_router
except ImportError:
    from mcp_client import start_mcp_client, stop_mcp_client
    from query_router import router as analytics_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_mcp_client()
    try:
        yield
    finally:
        await stop_mcp_client()


app = FastAPI(title="Fintech Analytics API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include the modular analytics and chat routes
app.include_router(analytics_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
