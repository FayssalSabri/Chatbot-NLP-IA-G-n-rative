FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download fr_core_news_sm

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "src/api/flask_app.py"]
