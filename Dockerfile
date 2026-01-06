# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and source code
COPY models/ ./models/
COPY face_detector.py ./
COPY main.py ./

# Expose port (if running a web server, adjust as needed)
# EXPOSE 8080

# Set the default command (adjust as needed)
CMD ["python", "main.py"]
