"""Hugging Face Space Entrypoint for SatQuery AI."""
import os
from deployment.config import deploy_settings
from deployment.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", deploy_settings.port))
    host = os.environ.get("HOST", deploy_settings.host)
    app.launch(
        server_name=host,
        server_port=port,
        prevent_thread_lock=False,
    )

