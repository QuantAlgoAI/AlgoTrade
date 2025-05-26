FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and TA-Lib
RUN apt-get update && apt-get install -y \
    gcc \
    wget \
    build-essential \
    && wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb \
    && dpkg -i ta-lib_0.6.4_amd64.deb \
    && rm ta-lib_0.6.4_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir logzero && \
    pip list

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs data_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata

# Run the application with default instrument (NIFTY 50)
CMD ["python", "AlgoTrade.py", "1"] 