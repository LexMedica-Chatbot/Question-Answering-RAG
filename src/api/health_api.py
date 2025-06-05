"""
Simple Health Check API
API sederhana untuk testing tanpa dependency eksternal
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
from datetime import datetime
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Health Check API",
    description="API sederhana untuk health check dan testing",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    message: str


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    original_message: str
    echo_message: str
    timestamp: str


# Global variable untuk tracking start time
start_time = time.time()


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "service": "Question Answering RAG System",
        "status": "running",
        "endpoints": ["/health", "/echo", "/docs", "/redoc"],
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    current_time = time.time()
    uptime = current_time - start_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2),
        message="API is running successfully",
    )


@app.post("/echo", response_model=EchoResponse)
async def echo_message(request: EchoRequest):
    """Echo endpoint untuk testing"""
    return EchoResponse(
        original_message=request.message,
        echo_message=f"Echo: {request.message}",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/info")
async def api_info():
    """API information endpoint"""
    return {
        "api_name": "Health Check API",
        "version": "1.0.0",
        "description": "Simple API untuk testing struktur project",
        "python_path": __file__,
        "available_endpoints": {
            "GET /": "Root endpoint",
            "GET /health": "Health check",
            "POST /echo": "Echo message",
            "GET /info": "API information",
            "GET /docs": "API documentation",
            "GET /redoc": "ReDoc documentation",
        },
    }


def main():
    """Main function untuk menjalankan API"""
    print("Starting Health Check API...")
    print(f"API will be available at: http://localhost:8000")
    print(f"Documentation at: http://localhost:8000/docs")
    uvicorn.run("health_api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
