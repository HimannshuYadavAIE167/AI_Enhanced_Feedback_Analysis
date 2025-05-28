# absa_model/sentiment_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Tuple, Union
import logging
import math

from config import ABSA_MODEL_NAME, ABSA_MODEL_TYPE, DEFAULT_DEVICE, INFERENCE_BATCH_SIZE

logger = logging.getLogger(__name__)

class AbsaAnalyzer:
    """
    Handles loading the ABSA model and performing sentiment predictions.
    Supports different model types based on config.
    """
    def __init__(self, model_name: str = ABSA_MODEL_NAME, model_type: str = ABSA_MODEL_TYPE, device: str = DEFAULT_DEVICE):
        self.model_name = model_name
        self.model_type = model_type
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _get_device(self, requested_device: str) -> torch.device:
        """Checks for CUDA availability and sets the device."""
        if requested_device == "cuda" and torch.cuda.is_available():
            logger.info("CUDA device found. Using GPU.")
            return torch.device("cuda")
        elif requested_device == "cuda":
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        else:
            logger.info("Using CPU device.")
            return torch.device("cpu")

    def _load_model(self):
        """Loads the tokenizer and model based on the configured type."""
        logger.info(f"Loading ABSA model: {self.model_name} (Type: {self.model_type})")
        try:
            if self.model_type == "sentence_pair":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval() # Set model to evaluation mode
                logger.info(f"Sentence-pair model '{self.model_name}' loaded successfully on {self.device}.")
            # Add elif block here for 'pyabsa' if implementing PyABSA support
            # elif self.model_type == "pyabsa":
            #     from pyabsa import AspectPolarityClassification as APC
            #     # Example: Load PyABSA model (adjust checkpoint as needed)
            #     self.model = APC.AspectExtractor(checkpoint=self.model_name, auto_device=True) # PyABSA handles device
            #     logger.info(f"PyABSA model '{self.model_name}' loaded successfully.")
            #     # Note: PyABSA might not use a separate tokenizer in the same way
            else:
                raise ValueError(f"Unsupported ABSA_MODEL_TYPE: {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _map_sentiment(self, prediction: int) -> str:
        """Maps model output label index to sentiment string."""
        # This mapping depends HEAVILY on the specific model's training labels.
        # Inspect the model's config (model.config.id2label or similar) on Hugging Face Hub.
        # Example mapping for a typical 3-class (neg, neu, pos) model:
        # Common labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        # Adjust based on model.config.id2label if available
        label_map = getattr(self.model.config, 'id2label', None)
        if label_map:
            try:
                # Standardize output labels
                sentiment = label_map.get(prediction, "Unknown").capitalize()
                if sentiment == "Negative": return "Negative"
                if sentiment == "Positive": return "Positive"
                if sentiment == "Neutral": return "Neutral"
                # Handle potential 'Conflict' label if model supports it [7, 26]
                if sentiment == "Conflict":
                    logger.debug("Conflict sentiment detected, mapping to Neutral.")
                    return "Neutral" # Or handle as a separate category if needed downstream
                return "Neutral" # Default fallback
            except Exception as e:
                 logger.warning(f"Could not map prediction {prediction} using model config: {e}. Falling back.")

        # Fallback mapping if config is unavailable or unclear
        # THIS IS AN ASSUMPTION - VERIFY FOR YOUR MODEL
        sentiment_map = {
            0: "Negative", # Often index 0 for negative
            1: "Neutral",  # Often index 1 for neutral
            2: "Positive", # Often index 2 for positive
            # 3: "Conflict" # If model has 4 labels [7, 26], map appropriately
        }
        # Map conflict to Neutral as a fallback if present
        if prediction == 3:
             logger.debug("Conflict sentiment prediction (fallback), mapping to Neutral.")
             return "Neutral"

        return sentiment_map.get(prediction, "Unknown")

    def predict_sentence_pairs(self, sentence_pairs: List[Tuple[str, str]]) -> List[Dict[str, Union[str, float]]]:
        """
        Performs batch prediction for sentence-pair classification models.

        Args:
            sentence_pairs (List]): A list of (sentence, aspect_query) tuples.

        Returns:
            List[Dict[str, Union[str, float]]]: List of dictionaries, each containing
                                                 'sentiment' (str) and 'confidence' (float).
                                                 Order matches the input list.
        """
        if self.model_type!= "sentence_pair":
            raise NotImplementedError("This prediction method is for sentence_pair models.")
        if not sentence_pairs:
            return

        results =[]
        num_batches = math.ceil(len(sentence_pairs) / INFERENCE_BATCH_SIZE)

        with torch.no_grad(): # Disable gradient calculations for inference
            for i in range(num_batches):
                batch_start = i * INFERENCE_BATCH_SIZE
                batch_end = batch_start + INFERENCE_BATCH_SIZE
                batch_pairs = sentence_pairs[batch_start:batch_end]

                # Separate sentences and queries for tokenization
                sentences1 = [pair for pair in batch_pairs]
                sentences2 = [pair for pair in batch_pairs]

                try:
                    inputs = self.tokenizer(sentences1, sentences2, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move inputs to the correct device

                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)

                    # Process results for the batch
                    for j in range(len(batch_pairs)):
                        prediction_idx = predictions[j].item()
                        sentiment = self._map_sentiment(prediction_idx)
                        confidence = probabilities[j, prediction_idx].item()
                        results.append({"sentiment": sentiment, "confidence": confidence})

                except Exception as e:
                    logger.error(f"Error during inference for batch {i+1}: {e}")
                    # Add placeholder results for the failed batch items
                    results.extend([{"sentiment": "Error", "confidence": 0.0}] * len(batch_pairs))

        return results

    # Add predict method for PyABSA if implementing
    # def predict_pyabsa(self, texts: List[str]) -> List[Dict[str, any]]:
    #     if self.model_type!= "pyabsa":
    #         raise NotImplementedError("This prediction method is for pyabsa models.")
    #     if not texts:
    #         return
    #     try:
    #         # PyABSA's extract_aspect handles batching and returns structured results
    #         # The format might differ, adjust parsing accordingly [2, 4]
    #         results = self.model.extract_aspect(inference_source=texts, pred_sentiment=True)
    #         # Process PyABSA results into a consistent format if needed
    #         processed_results =
    #         for res in results:
    #              # Example processing - structure may vary based on PyABSA version/task
    #              aspects_sentiments =
    #              if 'aspect' in res and 'sentiment' in res:
    #                   for aspect, sentiment in zip(res['aspect'], res['sentiment']):
    #                        aspects_sentiments.append({'aspect': aspect, 'sentiment': sentiment.capitalize()}) # Add confidence if available
    #              processed_results.append({'review_results': aspects_sentiments, 'original_sentence': res.get('sentence', '')})
    #         return processed_results
    #     except Exception as e:
    #         logger.error(f"Error during PyABSA inference: {e}")
    #         return [{"review_results":, "error": str(e)}] * len(texts)