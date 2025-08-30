.PHONY: install build up down logs clean test

# Install dependencies
install:
	pip install -r requirements.txt
	python -m spacy download fr_core_news_sm

# Build Docker images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Run tests
test:
	python -m pytest tests/

# Setup development environment
setup-dev: install
	pre-commit install
	mkdir -p data/raw data/processed models

# Generate sample data
sample-data:
	python scripts/generate_sample_data.py
