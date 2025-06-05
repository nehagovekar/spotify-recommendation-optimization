# Makefile for Spotify Hit Predictor & A/B Testing Platform

.PHONY: help install install-dev setup test lint format clean train analyze dashboard api docs

# Default target
help:
	@echo "🎵 Spotify Hit Predictor & A/B Testing Platform"
	@echo "================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Complete project setup (environment + dependencies)"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  test           - Run all tests with coverage"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black"
	@echo "  clean          - Clean temporary files and cache"
	@echo "  train          - Train ML models"
	@echo "  analyze        - Run A/B test analysis"
	@echo "  dashboard      - Launch Streamlit dashboard"
	@echo "  api            - Start FastAPI server"
	@echo "  docs           - Generate documentation"
	@echo "  pipeline       - Run complete ML pipeline"

# Environment setup
setup: clean
	@echo "🚀 Setting up development environment..."
	python -m venv venv
	@echo "✅ Virtual environment created"
	@echo "📦 Installing dependencies..."
	$(MAKE) install-dev
	@echo "🔧 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Setup complete!"
	@echo ""
	@echo "To activate the environment:"
	@echo "  source venv/bin/activate  # On macOS/Linux"
	@echo "  venv\\Scripts\\activate     # On Windows"

# Dependency installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Code quality
test:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running code linting..."
	pylint src/
	flake8 src/
	mypy src/

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# ML Pipeline commands
train:
	@echo "🤖 Training ML models..."
	python scripts/train_models.py

analyze:
	@echo "🧪 Running A/B test analysis..."
	python scripts/run_ab_test.py

pipeline:
	@echo "⚡ Running complete ML pipeline..."
	python scripts/run_full_pipeline.py

# Application deployment
dashboard:
	@echo "📊 Launching Streamlit dashboard..."
	streamlit run app/streamlit_dashboard.py

api:
	@echo "🚀 Starting FastAPI server..."
	uvicorn app.recommendation_api:app --reload --host 0.0.0.0 --port 8000

# Documentation
docs:
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/

# Data processing
download-data:
	@echo "📥 Downloading sample dataset..."
	mkdir -p data/raw
	# Add your data download commands here

process-data:
	@echo "🔄 Processing raw data..."
	python scripts/process_data.py

# Docker commands (if using containers)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t spotify-ml-platform .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 -p 8000:8000 spotify-ml-platform

# Git workflow helpers
commit-check:
	@echo "✅ Running pre-commit checks..."
	pre-commit run --all-files

push-ready: format lint test
	@echo "🚀 Ready to push!"
	git status

# Performance profiling
profile:
	@echo "⚡ Profiling model training..."
	python -m cProfile -o profile_results.prof scripts/train_models.py
	python -c "import pstats; pstats.Stats('profile_results.prof').sort_stats('cumulative').print_stats(20)"

# Database operations (if applicable)
init-db:
	@echo "🗄️ Initializing database..."
	python scripts/init_database.py

# Deployment helpers
deploy-local: clean install test
	@echo "🚀 Local deployment ready"

deploy-staging: clean install test lint
	@echo "🚀 Staging deployment ready"

deploy-prod: clean install test lint docs
	@echo "🚀 Production deployment ready"