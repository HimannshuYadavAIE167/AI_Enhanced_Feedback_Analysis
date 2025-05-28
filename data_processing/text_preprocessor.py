# data_processing/text_preprocessor.py
import re
import logging
from typing import List, Tuple, Optional

# Import based on configuration
from config import SENTENCE_SPLITTER_LIB

# Setup logging
logger = logging.getLogger(__name__)

if SENTENCE_SPLITTER_LIB == "spacy":
    try:
        import spacy
        # Load a small spaCy model for efficiency in sentence splitting
        # Ensure model is downloaded: python -m spacy download en_core_web_sm
        # Disable unused components like NER and lemmatizer if only sentence splitting is needed
        nlp_spacy = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        logger.info("Using spaCy for sentence splitting.")
    except ImportError:
        logger.error("spaCy not installed. Please install it: pip install spacy")
        raise # Re-raise the exception to stop execution if essential lib is missing
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Download it: python -m spacy download en_core_web_sm")
        raise # Re-raise the exception if model is not found
elif SENTENCE_SPLITTER_LIB == "nltk":
    try:
        import nltk
        # Ensure punkt tokenizer is downloaded: python -m nltk.downloader punkt
        nltk.data.find('tokenizers/punkt')
        logger.info("Using NLTK for sentence splitting.")
    except ImportError:
        logger.error("NLTK not installed. Please install it: pip install nltk")
        raise # Re-raise
    except LookupError:
        logger.error("NLTK 'punkt' tokenizer not found. Download it: python -m nltk.downloader punkt")
        raise # Re-raise
else:
    # Log an error and raise an exception for unsupported configuration
    logger.error(f"Unsupported SENTENCE_SPLITTER_LIB in config: {SENTENCE_SPLITTER_LIB}")
    raise ValueError(f"Unsupported SENTENCE_SPLITTER_LIB in config: {SENTENCE_SPLITTER_LIB}")


def clean_text(text: str) -> str:
    """
    Performs basic text cleaning.
    - Removes excessive whitespace.
    - Converts text to lowercase (required for uncased models).
    - (Add other cleaning steps as needed, e.g., handling special chars)
    """
    if not isinstance(text, str):
        logger.warning(f"clean_text received non-string input: {text}. Returning empty string.")
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    # --- FIX: Uncomment lowercasing for uncased models like DeBERTa base ---
    text = text.lower()
    # ---------------------------------------------------------------------
    return text

def split_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using the configured library.
    Returns a list of cleaned sentence strings.
    """
    if not text or not text.strip(): # Handle empty or whitespace-only input
        return []
    try:
        if SENTENCE_SPLITTER_LIB == "spacy":
            # Use the loaded spaCy pipeline
            doc = nlp_spacy(text)
            # Extract sentence texts and strip whitespace
            sentences = [sent.text.strip() for sent in doc.sents]
        elif SENTENCE_SPLITTER_LIB == "nltk":
            # Use NLTK's sentence tokenizer
            sentences = nltk.sent_tokenize(text)
            # Strip whitespace from each sentence
            sentences = [sent.strip() for sent in sentences]
        else:
            # Fallback if splitter is not configured correctly (should be caught by init block)
            logger.warning("No valid sentence splitter configured, returning text as single sentence.")
            sentences = [text.strip()] # Return original text as a single sentence

        # Filter out any empty strings that might result from splitting/stripping
        return [s for s in sentences if s]

    except Exception as e:
        # Log the error but return the original text in a list as a fallback
        logger.error(f"Error splitting sentences for text: '{text[:100]}...'. Error: {e}")
        return [text.strip()] # Return original text as fallback


def prepare_absa_input(sentence: str, aspect: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Formats input for sentence-pair classification ABSA models.
    Constructs an auxiliary sentence (query) based on the aspect.

    Args:
        sentence (str): The review sentence.
        aspect (str): The aspect term/category.

    Returns:
        Tuple[Optional[str], Optional[str]]: The original sentence and the constructed auxiliary sentence.
                                             Returns (None, None) if inputs are invalid.
    """
    # Ensure inputs are valid strings and not empty after stripping
    if not isinstance(sentence, str) or not sentence.strip() or \
       not isinstance(aspect, str) or not aspect.strip():
        logger.warning(f"Invalid input for prepare_absa_input: sentence='{sentence}', aspect='{aspect}'")
        return None, None

    # Use the cleaned sentence and aspect (which should now be lowercased)
    # The format "What is the sentiment of the aspect '...'?" is a common
    # query template for sentence-pair ABSA models.
    auxiliary_sentence = f"What is the sentiment of the aspect '{aspect.strip()}'?"

    # Return the two sequences. The Hugging Face tokenizer will handle
    # adding [CLS], [SEP] tokens and padding/truncation.
    return sentence.strip(), auxiliary_sentence

def extract_aspect_candidates(sentence: str, aspect_keywords: Optional[List[str]] = None) -> List[str]:
    """
    (Optional) Extracts potential aspect terms (noun phrases) from a sentence using spaCy.
    Can be refined using domain keywords or frequency analysis.

    Args:
        sentence (str): The input sentence.
        aspect_keywords (Optional[List[str]]): Optional list of keywords to filter aspects.

    Returns:
        List[str]: A list of potential aspect noun phrases. Returns empty list if spaCy
                   is not configured, input is invalid, or no candidates found.
    """
    # Check if spaCy is the configured splitter and if the nlp object was loaded
    if SENTENCE_SPLITTER_LIB != "spacy" or 'nlp_spacy' not in globals():
        # logger.warning("extract_aspect_candidates currently requires spaCy. Skipping.") # Avoid excessive logging
        return [] # Return empty list

    if not isinstance(sentence, str) or not sentence.strip():
        return []

    candidates = []
    try:
        # Process the sentence with the loaded spaCy model
        doc = nlp_spacy(sentence.strip())
        # Extract noun chunks as potential aspects
        for chunk in doc.noun_chunks:
            # Basic filtering: ignore pronouns, very short chunks, or chunks that are just punctuation/symbols
            chunk_text = chunk.text.strip()
            if chunk.root.pos_ != "PRON" and len(chunk_text) > 1 and not re.fullmatch(r'[^\w\s]+', chunk_text):
                # Normalize candidate by lowercasing
                candidate = chunk_text.lower()
                # Optional: Filter based on keywords if provided
                if aspect_keywords:
                    if any(keyword.lower() in candidate for keyword in aspect_keywords):
                        candidates.append(candidate)
                else:
                    candidates.append(candidate) # Add candidate if no keywords are provided

        # Return unique candidates
        return list(set(candidates))

    except Exception as e:
        logger.error(f"Error extracting aspect candidates from sentence: '{sentence[:100]}...'. Error: {e}")
        return [] # Return empty list on error


# Example usage (for testing independently)
if __name__ == "__main__":
    # Example of cleaning and splitting
    text = " This is the first sentence. And this is the second one!! It's great. "
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    print(f"Original: '{text}'")
    print(f"Cleaned: '{cleaned}'")
    print(f"Sentences: {sentences}")

    # Example of preparing ABSA input
    sentence_ex = "The food was absolutely delicious."
    aspect_ex = "food"
    sent1, sent2 = prepare_absa_input(sentence_ex, aspect_ex)
    print(f"ABSA Input Pair: ('{sent1}', '{sent2}')")

    # Example of extracting aspect candidates (requires spaCy)
    if SENTENCE_SPLITTER_LIB == "spacy" and 'nlp_spacy' in globals():
        sentence_aspect = "The service was slow, but the food was amazing."
        aspect_candidates = extract_aspect_candidates(sentence_aspect)
        print(f"Aspect Candidates for '{sentence_aspect}': {aspect_candidates}")

        sentence_aspect_filtered = "The service was slow, but the food was amazing."
        aspect_candidates_filtered = extract_aspect_candidates(sentence_aspect_filtered, aspect_keywords=["service", "food", "price"])
        print(f"Filtered Aspect Candidates for '{sentence_aspect_filtered}': {aspect_candidates_filtered}")
    else:
        print("SpaCy not available or configured for aspect extraction examples.")
