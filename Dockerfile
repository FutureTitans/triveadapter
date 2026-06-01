FROM python:3.11-slim

# Install system dependencies and Node.js for building frontend
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements first to leverage Docker cache
COPY backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install TRIBE v2 from GitHub (not on PyPI)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/tribev2.git

# Copy frontend package files
COPY frontend/package*.json ./frontend/

# Install Node dependencies
WORKDIR /app/frontend
RUN npm install

# Copy all project files
WORKDIR /app
COPY . .

# Build frontend
WORKDIR /app/frontend
RUN npm run build

# Setup backend for deployment
WORKDIR /app/backend

# Create cache directory for HuggingFace models
RUN mkdir -p cache

# Expose port (7860 is default for HuggingFace Spaces)
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
