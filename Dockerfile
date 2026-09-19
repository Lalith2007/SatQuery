FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Install system dependencies & Node.js for React build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy dependency specifications first for layer caching
COPY --chown=user:user pyproject.toml ./
COPY --chown=user:user frontend/package*.json ./frontend/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn spaces

# Build React frontend
RUN cd frontend && npm install && npm run build

# Copy remaining application source
COPY --chown=user:user . .

EXPOSE 7860

CMD ["python", "app.py"]
