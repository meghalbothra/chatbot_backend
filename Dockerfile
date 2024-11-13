# Use the official Python image
FROM python:3.9-slim

# Install system dependencies required by psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port the app will run on
EXPOSE 8000

# Command to run your application (adjust accordingly for your app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
