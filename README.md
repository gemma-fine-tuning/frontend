# Gemma Fine-Tuning Frontend

This is the Streamlit frontend for the Gemma Fine-Tuning Platform. It provides a user-friendly interface for managing the fine-tuning workflow.

## Features

- Dataset selection and upload
- Data preprocessing configuration
- Model configuration settings
- Training monitoring and results

## Installation

> Please note that though the commands given use `uv`, it's because we use `uv` to develop the application. You can use equivalent `pip` commands and the project will work just fine.

```bash
uv venv
```

## Running the App

```bash
uv run streamlit run main.py
```

## Deployment

This frontend is designed to be deployed on Hugging Face Spaces. Simply push this directory to a Hugging Face Space repository.

## Configuration

The app connects to backend services for:

- Data preprocessing (CPU-based Cloud Run service)
- Model fine-tuning (GPU-based Cloud Run service)

Backend endpoints can be configured via environment variables or Streamlit secrets.
