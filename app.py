"""Hugging Face Space Entrypoint for SatQuery AI."""
import os
import uvicorn
from deployment.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
