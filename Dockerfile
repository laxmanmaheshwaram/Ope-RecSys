# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (needed for numerical libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and project files
COPY . .

# Run the experiment by default
CMD ["python", "main.py"]