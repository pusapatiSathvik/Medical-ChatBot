# Use a slim version of Python 3.10 to keep the image size small
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for certain AI libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the application
# Using gunicorn for production instead of the Flask dev server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]