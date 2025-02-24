# Base image
FROM python:3.10-slim

# Install ffmpeg and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot script and other project files
COPY . .

# Ensure logs are shown in real-time
ENV PYTHONUNBUFFERED=1

# Start the bot
CMD ["python", "bot.py"]
