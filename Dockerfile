# Use a lightweight, official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project code into the container
COPY . .

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "entrypoint/app.py", "--server.port=8501", "--server.address=0.0.0.0"]