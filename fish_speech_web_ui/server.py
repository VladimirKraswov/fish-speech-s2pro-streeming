import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="Fish Speech Web UI", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint (must be before mounting static files)
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "web-ui"})

# Path to static files
STATIC_DIR = Path(__file__).parent / "static"

def setup_static(app: FastAPI):
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:
        @app.get("/")
        async def root():
            return JSONResponse({
                "message": "Web UI server is running, but static assets are missing. Please run the build script.",
                "expected_path": str(STATIC_DIR.absolute())
            }, status_code=404)

setup_static(app)

def main():
    parser = argparse.ArgumentParser(description="Fish Speech Web UI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=9001, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
