# absa_model/batch_inference.py
import logging
import time
from typing import List, Dict, Any

# Project modules
import config
from data_processing.yelp_loader import load_yelp_reviews
from data_processing.text_preprocessor import clean_text, split_sentences, prepare_absa_input, extract_aspect_candidates
from absa_model.sentiment_analyzer import AbsaAnalyzer
from database.mongo_client import MongoClientWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def structure_results_for_db(review_batch: List[Dict[str, Any]], all_predictions: List[Dict[str, Any]], sentence_map: List[Any]) -> List[Dict[str, Any]]:
    """
    Structures the raw predictions into the desired format for MongoDB insertion.

    Args:
        review_batch (List[Dict[str, Any]]): The batch of original review data.
        all_predictions (List[Dict[str, Any]]): Flat list of predictions from AbsaAnalyzer.
        sentence_map (List]): List mapping each prediction back to its
                                                  (review_index_in_batch, sentence_index_in_review, aspect).

    Returns:
        List[Dict[str, Any]]: List of documents ready for MongoDB insertion.
    """
    results_for_db =[]
    processed_predictions = 0

    # Create a structure to hold results per review
    review_results_map = {i: {'review_id': review['review_id'],
                              'original_text': review['text'],
                              'stars': review.get('stars'),
                              'analysis_results': []}
                          for i, review in enumerate(review_batch)}

    if len(all_predictions)!= len(sentence_map):
         logger.error(f"Mismatch between predictions ({len(all_predictions)}) and sentence map ({len(sentence_map)}). Results may be incorrect.")
         # Handle error case, maybe return empty or partial results
         return

    for i, prediction in enumerate(all_predictions):
        try:
            review_idx, sentence_idx, aspect = sentence_map[i]
            # Find the original sentence text (requires storing sentences during processing)
            # This part needs refinement - assumes sentences were stored or can be retrieved
            # For simplicity, we'll assume sentences were processed sequentially and can be mapped back
            # A more robust approach would store sentence text alongside the map
            original_sentence = "Sentence text not available in this structure" # Placeholder

            analysis_entry = {
                "sentence_index": sentence_idx, # Index within the review's sentences
                # "sentence_text": original_sentence, # Store sentence for context
                "aspect": aspect,
                "sentiment": prediction.get("sentiment", "Unknown"),
                "confidence": prediction.get("confidence", 0.0)
                # Add span indices if available/extracted
            }
            review_results_map[review_idx]['analysis_results'].append(analysis_entry)
            processed_predictions += 1
        except IndexError:
             logger.error(f"IndexError accessing sentence_map at index {i}. Skipping prediction.")
        except KeyError:
             logger.error(f"KeyError accessing review_results_map with review_idx {review_idx}. Skipping prediction.")
        except Exception as e:
             logger.error(f"Unexpected error structuring prediction {i}: {e}. Skipping.")


    logger.info(f"Structured {processed_predictions}/{len(all_predictions)} predictions for DB.")
    return list(review_results_map.values())


def run_batch_inference():
    """
    Runs the batch inference pipeline: Load -> Preprocess -> Predict -> Store.
    """
    logger.info("Starting batch inference process...")
    start_time = time.time()

    # Initialize components
    try:
        analyzer = AbsaAnalyzer()
        mongo_client = MongoClientWrapper(
            uri=config.MONGO_URI,
            db_name=config.MONGO_DB_NAME,
            collection_name=config.MONGO_COLLECTION_NAME
        )
        # Ensure indexes exist
        mongo_client.create_indexes()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return

    total_reviews_processed = 0
    total_pairs_processed = 0
    batch_num = 0

    # Process data in batches from the generator
    for review_batch in load_yelp_reviews(config.YELP_DATA_PATH, batch_size=config.INFERENCE_BATCH_SIZE):
        batch_num += 1
        logger.info(f"Processing batch {batch_num} with {len(review_batch)} reviews.")
        batch_start_time = time.time()

        sentence_pairs_for_batch =[]
        sentence_map = []# Maps each pair back to (review_idx, sentence_idx, aspect)

        for review_idx, review in enumerate(review_batch):
            try:
                cleaned_text = clean_text(review['text'])
                sentences = split_sentences(cleaned_text)

                # --- Aspect Handling ---
                # Strategy 1: Use predefined aspects for sentence-pair model
                aspects_to_analyze = config.PREDEFINED_ASPECTS

                # Strategy 2: Extract candidates dynamically (if using text_preprocessor.extract_aspect_candidates)
                # if not aspects_to_analyze:
                #     all_candidates =
                #     for sent in sentences:
                #         all_candidates.extend(extract_aspect_candidates(sent))
                #     aspects_to_analyze = list(set(all_candidates)) # Use unique candidates found in the review

                if not aspects_to_analyze:
                     logger.debug(f"No aspects to analyze for review {review['review_id']}. Skipping.")
                     continue

                for sent_idx, sentence in enumerate(sentences):
                    if not sentence: continue
                    for aspect in aspects_to_analyze:
                        # Prepare input for the specific model type
                        if config.ABSA_MODEL_TYPE == "sentence_pair":
                            sentence1, sentence2 = prepare_absa_input(sentence, aspect)
                            if sentence1 and sentence2:
                                sentence_pairs_for_batch.append((sentence1, sentence2))
                                sentence_map.append((review_idx, sent_idx, aspect))
                        # Add elif for 'pyabsa' input preparation if needed
                        # elif config.ABSA_MODEL_TYPE == "pyabsa":
                        #    # PyABSA might take raw sentences directly
                        #    # Need to structure input and map results differently
                        #    pass

            except Exception as e:
                logger.error(f"Error processing review {review.get('review_id', 'N/A')}: {e}")
                # Continue to next review in batch

        # Perform inference on the collected pairs for the batch
        if sentence_pairs_for_batch:
            logger.info(f"Running inference on {len(sentence_pairs_for_batch)} sentence-aspect pairs for batch {batch_num}.")
            try:
                predictions = analyzer.predict_sentence_pairs(sentence_pairs_for_batch)
                total_pairs_processed += len(sentence_pairs_for_batch)

                # Structure results for DB insertion
                db_documents = structure_results_for_db(review_batch, predictions, sentence_map)

                # Insert results into MongoDB
                if db_documents:
                    mongo_client.insert_results(db_documents)
                    logger.info(f"Inserted results for batch {batch_num} into MongoDB.")
                else:
                     logger.warning(f"No documents generated for database insertion in batch {batch_num}.")

            except Exception as e:
                logger.error(f"Error during inference or DB insertion for batch {batch_num}: {e}")
        else:
            logger.info(f"No sentence-aspect pairs generated for batch {batch_num}.")

        total_reviews_processed += len(review_batch)
        batch_end_time = time.time()
        logger.info(f"Batch {batch_num} processed in {batch_end_time - batch_start_time:.2f} seconds.")

    end_time = time.time()
    logger.info(f"Batch inference process completed.")
    logger.info(f"Total reviews processed: {total_reviews_processed}")
    logger.info(f"Total sentence-aspect pairs processed: {total_pairs_processed}")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_batch_inference()