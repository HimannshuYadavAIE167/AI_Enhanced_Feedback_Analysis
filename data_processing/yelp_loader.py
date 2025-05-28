# data_processing/yelp_loader.py
import jsonlines
import logging
from typing import Iterator, List, Dict, Any

logger = logging.getLogger(__name__)

def load_yelp_reviews(file_path: str, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
    """
    Loads Yelp reviews from a JSONL file in batches using a generator.

    Args:
        file_path (str): The path to the Yelp dataset JSONL file.
        batch_size (int): The number of reviews to yield in each batch.

    Yields:
        Iterator[List[Dict[str, Any]]]: An iterator yielding batches of review dictionaries.
                                         Each dictionary should contain at least
                                         'review_id', 'text', and 'stars'.
    """
    logger.info(f"Starting to load Yelp reviews from {file_path} in batches of {batch_size}")
    batch = []
    try:
        with jsonlines.open(file_path, mode='r') as reader:
            for i, review in enumerate(reader):
                # Basic validation: ensure essential keys exist
                if 'review_id' in review and 'text' in review and 'stars' in review:
                    # Select only necessary fields to conserve memory if needed
                    processed_review = {
                        'review_id': review['review_id'],
                        'text': review['text'],
                        'stars': review.get('stars', None) # Use.get for optional fields
                    }
                    batch.append(processed_review)
                    if len(batch) >= batch_size:
                        logger.debug(f"Yielding batch {i // batch_size + 1}")
                        yield batch
                        batch = []
                else:
                    logger.warning(f"Skipping review at line {i+1} due to missing essential keys (review_id, text, stars).")

            # Yield any remaining reviews in the last batch
            if batch:
                logger.debug("Yielding final batch.")
                yield batch
        logger.info("Finished loading Yelp reviews.")
    except FileNotFoundError:
        logger.error(f"Error: Yelp data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while reading {file_path}: {e}")
        raise