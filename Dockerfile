FROM python:3.11-slim

# Force CPU-only mode at the OS level, BEFORE Python/PyTorch starts.
# This makes torch.cuda.is_available() return False globally.
ENV CUDA_VISIBLE_DEVICES=""

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (provides uvx command, required by TRIBE v2 for whisperx)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TRIBE v2 from GitHub (not on PyPI)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/tribev2.git

# Copy all backend files
COPY . .

# Create cache directory for HuggingFace models
RUN mkdir -p cache

# Expose port
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
