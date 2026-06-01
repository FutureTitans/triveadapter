FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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
