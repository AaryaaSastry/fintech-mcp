from fastapi import FastAPI
from .query_router import router as analytics_router
import uvicorn

app = FastAPI(title="Fintech Analytics API")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include the modular analytics and chat routes
app.include_router(analytics_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
