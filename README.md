# AI-Powered Aspect-Based Sentiment Analysis Dashboard for Customer Reviews

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Setup and Installation](#setup-and-installation)
   - [Prerequisites](#prerequisites)
   - [Cloning the Repository](#cloning-the-repository)
   - [Setting up the Environment](#setting-up-the-environment)
   - [Configuration](#configuration)
   - [Database Setup](#database-setup)
6. [Usage](#usage)
   - [Running the Batch Inference Pipeline](#running-the-batch-inference-pipeline)
   - [Starting the Dashboard](#starting-the-dashboard)
7. [Dashboard Overview](#dashboard-overview)


## Introduction

This project implements an AI-powered system for performing Aspect-Based Sentiment Analysis (ABSA) on customer reviews (e.g., Yelp reviews). It extracts specific aspects mentioned in the reviews and determines the sentiment expressed towards each aspect. The results are stored in a MongoDB database and visualized through an interactive Streamlit dashboard, providing actionable insights into customer feedback.

The core idea is to move beyond general sentiment analysis (positive/negative/neutral for the whole review) and understand *what* specific things customers like or dislike.

## Features

* **Data Ingestion:** Processes customer review data (e.g., from CSV or JSON files).
* **Text Preprocessing:** Cleans and prepares review text for analysis.
* **Aspect Extraction:** Identifies key aspects or topics discussed in reviews (e.g., "food," "service," "ambiance," "price").
* **Sentiment Analysis:** Assigns sentiment (Positive, Negative, Neutral) to each identified aspect using a pre-trained transformer model (e.g., `yangheng/deberta-v3-base-absa-v1.1` or another model from Hugging Face).
* **Database Storage:** Stores raw reviews and structured analysis results (aspects, sentiments, confidence scores) in MongoDB for efficient querying.
* **Interactive Dashboard:** A Streamlit application to:
  * Display overall sentiment distribution.
  * Show sentiment breakdown per aspect.
  * Filter reviews by aspect, sentiment, or star rating.
  * View individual reviews with their corresponding aspect sentiments.
  * Provide summary statistics.

## Tech Stack

* **Programming Language:** Python 3.x
* **Core Libraries:**
  * `transformers` (Hugging Face): For ABSA models.
  * `torch` or `tensorflow`: Backend for transformer models.
  * `pandas`: For data manipulation.
  * `spacy` (Optional, if used for advanced text processing/aspect extraction).
  * `nltk` (Optional, if used for text processing).
* **Database:** MongoDB
  * `pymongo`: Python driver for MongoDB.
* **Dashboard:** Streamlit
  * `plotly` (Optional, for advanced charts in Streamlit).
* **Environment Management:** `venv` (or Conda)
* **Configuration:** `python-dotenv` (for managing environment variables).

## Project Structure

```
.
├── absa_model/                 # Core ABSA logic, sentiment analyzer
│   ├── __init__.py
│   ├── sentiment_analyzer.py   # Class for loading model and prediction
│   └── batch_inference.py      # Script to run inference on dataset
├── data_processing/            # Scripts for data loading and preprocessing
│   ├── __init__.py
│   └── text_preprocessor.py    # Functions for cleaning text, aspect extraction helpers
├── database/                   # MongoDB interaction logic
│   ├── __init__.py
│   └── mongo_client.py         # Wrapper for MongoDB connections and operations
├── dashboard/                  # Streamlit dashboard application
│   ├── __init__.py
│   ├── app.py                  # Main Streamlit application script
│   └── utils.py                # Utility functions for the dashboard
├── data/                       # (Optional) Raw data files (ensure .gitignore if large/sensitive)
│   └── reviews.csv
├── tests/                      # (Optional) Unit and integration tests
├── .env.example                # Example environment variables file
├── .gitignore                  # Specifies intentionally untracked files
├── config.py                   # Project configuration (model names, paths, etc.)
├── main.py                     # Main script to orchestrate pipeline or run components
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## Setup and Installation

### Prerequisites

* Python 3.8+
* Git
* MongoDB installed and running. You can get it from [MongoDB's official website](https://www.mongodb.com/try/download/community).
* (Optional) If using GPU for inference: NVIDIA drivers, CUDA, and cuDNN compatible with your PyTorch/TensorFlow version.

### Cloning the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <your-repository-name>
```

### Setting up the Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you encountered specific library version issues during development, ensure `requirements.txt` reflects those working versions).*

### Configuration

1. **Environment Variables:**
   Rename `.env.example` to `.env` and update it with your specific configurations:

   ```env
   # .env
   MONGO_URI="mongodb://localhost:27017/"
   MONGO_DB_NAME="yelp_absa_db" # Or your preferred database name
   MONGO_COLLECTION_NAME="review_sentiments" # Or your preferred collection name

   # Optional: Hugging Face model cache directory if you want to override the default
   # HF_HOME="/path/to/your/huggingface_cache"
   ```

2. **Model Configuration (`config.py`):**
   Review `config.py` and ensure settings like `ABSA_MODEL_NAME`, `DEFAULT_DEVICE` (cpu/cuda), `INFERENCE_BATCH_SIZE` are appropriate for your setup.
   
   Example `config.py` might look like:

   ```python
   # config.py
   import os
   from dotenv import load_dotenv

   load_dotenv() # Load variables from .env file

   # --- Model Configuration ---
   ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
   # ABSA_MODEL_NAME = "lhoestq/distilbert-base-uncased-finetuned-absa-as" # Alternative
   ABSA_MODEL_TYPE = "sentence_pair" # or "pyabsa" if using that library directly

   # --- Device Configuration ---
   DEFAULT_DEVICE = "cpu" # "cuda" if GPU is available and configured

   # --- Inference Configuration ---
   INFERENCE_BATCH_SIZE = 16

   # --- Data Paths (Update as needed) ---
   RAW_DATA_PATH = "data/your_reviews.csv"
   PROCESSED_DATA_PATH = "data/processed_reviews.json"

   # --- MongoDB Configuration ---
   MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
   MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "yelp_absa_db")
   MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "review_sentiments")

   # --- Aspect Configuration (if using predefined aspects) ---
   PREDEFINED_ASPECTS = [
       "food", "service", "price", "ambiance", "cleanliness", "location", "drinks"
       # Add or modify aspects relevant to your dataset
   ]

   # --- Logging Configuration ---
   LOG_LEVEL = "INFO"
   LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   ```

### Database Setup

* Ensure your MongoDB server is running. The database and collection specified in your configuration (`.env` or `config.py`) will typically be created automatically by the application when data is first inserted if they don't already exist.
* (Optional) You might want to create database indexes for better query performance after initial data insertion. The `mongo_client.py` might include a function for this.

## Usage

### Running the Batch Inference Pipeline

This step processes your raw review data, performs ABSA, and stores the results in MongoDB.

1. **Prepare your data:** Ensure your raw review data (e.g., `your_reviews.csv`) is in the location specified in `config.py` (e.g., `data/` directory). The script should expect a certain format (e.g., columns like `review_id`, `text`, `stars`).

2. **Run the main inference script:**
   (The exact command might vary based on your `main.py` or specific batch inference script)

   ```bash
   python main.py --run-inference
   ```

   or if you have a direct script:

   ```bash
   python absa_model/batch_inference.py --input_file data/your_reviews.csv
   ```

   Check the script's arguments or `main.py` for precise commands.

   The first time you run this, the Hugging Face model will be downloaded. This might take some time depending on your internet connection and the model size.

### Starting the Dashboard

Once the database is populated with analysis results:

1. **Navigate to the dashboard directory (if `app.py` is there):**
   ```bash
   # If your streamlit app.py is in the root directory
   streamlit run app.py

   # Or, if it's inside the dashboard/ folder
   streamlit run dashboard/app.py
   ```

2. Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Dashboard Overview

The dashboard provides several views:

* **Summary View:** Overall sentiment distribution, total reviews analyzed.
* **Aspect View:** Select an aspect (e.g., "food") to see its specific sentiment breakdown (e.g., 70% Positive, 20% Negative, 10% Neutral for "food").
* **Review Explorer:** Filter reviews by aspect, sentiment, star rating, and view the review text along with the extracted aspect sentiments.
Example Screenshots:

Main Dashboard View: 
Aspect Sentiment Detail: 

