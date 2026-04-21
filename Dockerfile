# Use a lightweight, official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and general building
# Replace the multi-line apt-get command with this single line
# We swapped the deprecated package for the modern libgl1 and libglib2.0-0
RUN apt-get update --fix-missing && apt-get install -y build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*


# Copy requirements and install the rest of your packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run your app (with the fileWatcher turned off to stop that terminal error!)
CMD ["streamlit", "run", "entrypoint/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]