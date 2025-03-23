# Use an official Python image as a base
FROM python:3.11

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages (MySQL client)
RUN apt-get update && \
    apt-get install -y default-mysql-client && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy application files into the container
COPY . .

# Install Python dependencies (Streamlit, MySQL connector, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit UI
CMD ["streamlit", "run", "UI.py", "--server.port=8501", "--server.address=0.0.0.0"]
