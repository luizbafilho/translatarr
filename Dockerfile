# Use Python 3.12 as base image
FROM python:3.12-slim

# Accept build arguments
ARG USER_ID=1026
ARG GROUP_ID=1000
ARG TZ=UTC

# Install FFmpeg and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create group and user with provided IDs
RUN if getent group $GROUP_ID > /dev/null 2>&1; then \
        groupmod -n appuser $(getent group $GROUP_ID | cut -d: -f1); \
    else \
        groupadd -g $GROUP_ID appuser; \
    fi && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set ownership of the application files
RUN chown -R appuser:appuser /app

# Environment variables
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Run the service
CMD ["python", "main.py"]