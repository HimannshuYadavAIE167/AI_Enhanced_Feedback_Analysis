# config.py
import os
from dotenv import load_dotenv

# Load environment variables from.env file if it exists
load_dotenv()

# --- Project Root ---
# Assumes config.py is at the project root
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Data ---
# Assumes data is in a 'data' subdirectory relative to project root
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
YELP_DATA_FILENAME = "yelp_academic_dataset_review.json" # Assuming JSON Lines format for efficiency
YELP_DATA_PATH = os.path.join(DATA_DIR, YELP_DATA_FILENAME)

# --- Model ---
# Option 1: Using a sentence-pair classification model fine-tuned for ABSA
# This model expects (sentence, aspect_query) pairs as input.
# ABSA_MODEL_NAME = "lhoestq/distilbert-base-uncased-finetuned-absa-as" 
ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"# [3]
# Option 2: Using PyABSA (requires different implementation in absa_model/sentiment_analyzer.py)
# ABSA_MODEL_NAME = "english" # Or specific PyABSA checkpoint [2, 4]
# ABSA_MODEL_TYPE = "pyabsa" # Flag to indicate PyABSA usage
ABSA_MODEL_TYPE = "sentence_pair" # Flag for the Hugging Face model type

INFERENCE_BATCH_SIZE = 16 # Adjust based on GPU memory and model size [5]
DEFAULT_DEVICE = "cuda" # Use "cuda" if GPU is available, otherwise "cpu"

# --- Database ---
# Use environment variable for URI, fallback to localhost default
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") # [1, 6]
MONGO_DB_NAME = "yelp_absa_db"
MONGO_COLLECTION_NAME = "review_sentiments"

# --- Aspects ---
# Predefined aspects for potential guidance or mapping, especially for restaurant domain.
# Useful if the ABSA model primarily classifies sentiment for given aspects,
# or for structuring dashboard filters.
# Uncomment this line and provide a list of aspects
PREDEFINED_ASPECTS = ["food", "service", "price", "ambiance", "wait time", "staff", "cleanliness", "menu"] 

# --- Text Processing ---
# Consider using spaCy or NLTK for sentence splitting
# Ensure the corresponding library is installed (e.g., pip install spacy && python -m spacy download en_core_web_sm)
SENTENCE_SPLITTER_LIB = "spacy" # Options: "spacy", "nltk"