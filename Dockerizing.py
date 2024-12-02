'''Dockerizing the Services
To containerize both the UI backend and the AI backend, Docker will be used.

Dockerfile for UI Backend (Flask Example):
# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port that the application runs on
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]

Dockerfile for AI Backend (Flask Example):


# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port that the application runs on
EXPOSE 5001

# Command to run the app
CMD ["python", "app.py"]

docker-compose.yml
To manage both services together in containers, Docker Compose can be used.'''


version: '3.8'

services:
  ui-backend:
    build:
      context: ./ui-backend
    ports:
      - "5000:5000"
    depends_on:
      - ai-backend

  ai-backend:
    build:
      context: ./ai-backend
    ports:
      - "5001:5001"
