FROM python:3.10-slim

WORKDIR /app

# Install Linux deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit needs this to run properly
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8000

# Command to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8000", "--server.address=0.0.0.0"]
