# database/mongo_client.py
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MongoClientWrapper:
    """
    A wrapper class for handling MongoDB connections and operations
    for the ABSA project.
    """
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        """Establishes connection to MongoDB."""
        try:
            # Timeout after 5 seconds for server selection
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            # It's a quick way to check if the connection is successful.
            self.client.admin.command('ismaster')
            # Get the database and collection objects
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"Successfully connected to MongoDB at {self.uri}, database '{self.db_name}', collection '{self.collection_name}'.")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB at {self.uri}: {e}")
            # Ensure objects are None on failure
            self.client = None
            self.db = None
            self.collection = None
            # Re-raise the exception so the calling code knows connection failed
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during MongoDB connection: {e}")
            # Ensure objects are None on failure
            self.client = None
            self.db = None
            self.collection = None
            # Re-raise the exception
            raise

    def close_connection(self):
        """Closes the MongoDB connection."""
        # Corrected check: Use 'is not None' for Pymongo client object
        if self.client is not None:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def create_indexes(self):
        """Creates necessary indexes for efficient querying."""
        # Corrected check: Use 'is None' for Pymongo collection object
        if self.collection is None:
            logger.error("Cannot create indexes: No MongoDB collection available.")
            return

        try:
            # Index on review_id for quick review lookup (ensure uniqueness)
            self.collection.create_index("review_id", unique=True, name="review_id_unique_idx")
            # Multikey index on aspect within the analysis_results array for filtering/aggregation
            self.collection.create_index("analysis_results.aspect", name="aspect_multikey_idx")
            # Compound multikey index for filtering by aspect and sentiment together
            self.collection.create_index([("analysis_results.aspect", 1), ("analysis_results.sentiment", 1)], name="aspect_sentiment_multikey_idx")
            # Index on stars for filtering by rating (sparse if 'stars' might be missing)
            self.collection.create_index("stars", name="stars_idx", sparse=True)

            logger.info("Ensured necessary MongoDB indexes exist.")
        except OperationFailure as e:
            logger.error(f"Failed to create indexes: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during index creation: {e}")


    def insert_results(self, results_list: List[Dict[str, Any]]):
        """
        Inserts a list of structured ABSA results into the collection.
        Uses update_one with upsert=True to avoid duplicates based on review_id.

        Args:
            results_list (List[Dict[str, Any]]): A list of documents, each structured
                                                 as defined in batch_inference.py.
        """
        # Corrected check: Use 'is None' for Pymongo collection object
        if self.collection is None:
            logger.error("Cannot insert results: No MongoDB collection available.")
            return
        if not results_list:
            logger.warning("No results provided to insert.")
            return

        operations = []
        for result in results_list:
            # Ensure each document has a review_id for the upsert operation
            if 'review_id' not in result:
                logger.warning(f"Skipping insertion for result missing 'review_id': {str(result)[:100]}...")
                continue
            # Use update_one with upsert=True to insert a new document if review_id
            # doesn't exist, or update the existing document if it does.
            operations.append(
                pymongo.UpdateOne(
                    {"review_id": result["review_id"]}, # Filter document by review_id
                    {"$set": result}, # Set the entire document content
                    upsert=True # Insert if not found
                )
            )

        # Check if any valid operations were created
        if not operations:
             logger.warning("No valid operations generated for bulk write.")
             return

        try:
            # Perform bulk write for efficiency. ordered=False allows other
            # operations to succeed even if one fails.
            bulk_result = self.collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk write completed. Matched: {bulk_result.matched_count}, Upserted: {bulk_result.upserted_count}, Modified: {bulk_result.modified_count}")
        except pymongo.errors.BulkWriteError as bwe:
            # Log details of bulk write errors
            logger.error(f"Bulk write error during insertion: {bwe.details}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during bulk insertion: {e}")

    def get_sentiments_by_aspect(self, aspect_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves sentiment distribution for a specific aspect using aggregation pipeline.

        Args:
            aspect_name (str): The aspect to query (case-sensitive, adjust query if needed).

        Returns:
            List[Dict[str, Any]]: List of aggregation results, e.g., [{'sentiment': 'Positive', 'count': 100},...].
                                 Returns empty list if collection is not available or on error.
        """
        # Corrected check: Use 'is None' for Pymongo collection object
        if self.collection is None:
            logger.error("Cannot get sentiments: No MongoDB collection available.")
            return [] # Return empty list as per type hint

        try:
            # Aggregation pipeline to unwind analysis_results, match the aspect,
            # group by sentiment, and count occurrences.
            pipeline = [
                {"$unwind": "$analysis_results"}, # Deconstruct the analysis_results array
                {"$match": {"analysis_results.aspect": aspect_name}}, # Filter documents by aspect name
                {"$group": {
                    "_id": "$analysis_results.sentiment", # Group results by sentiment value
                    "count": {"$sum": 1} # Count documents in each group
                }},
                {"$project": { # Reshape the output documents
                    "_id": 0, # Exclude the default _id field
                    "sentiment": "$_id", # Rename _id field to sentiment
                    "count": 1 # Include the count field
                }},
                {"$sort": {"sentiment": 1}} # Sort results alphabetically by sentiment
            ]
            results = list(self.collection.aggregate(pipeline))
            return results
        except Exception as e:
            logger.error(f"Error querying sentiments for aspect '{aspect_name}': {e}")
            return [] # Return empty list on error


    def get_reviews_by_sentiment(self, aspect_name: Optional[str] = None, sentiment: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves sample reviews matching aspect and/or sentiment criteria.

        Args:
            aspect_name (Optional[str]): The aspect to filter by.
            sentiment (Optional[str]): The sentiment to filter by.
            limit (int): Maximum number of reviews to return.

        Returns:
            List[Dict[str, Any]]: List of review documents matching the criteria.
                                 Returns empty list if collection is not available or on error.
        """
        # Corrected check: Use 'is None' for Pymongo collection object
        if self.collection is None:
            logger.error("Cannot get reviews: No MongoDB collection available.")
            return [] # Return empty list

        query = {}
        # Build the query using $elemMatch if filtering by aspect or sentiment
        if aspect_name or sentiment:
            elem_match_filter = {}
            if aspect_name:
                elem_match_filter["aspect"] = aspect_name
            if sentiment:
                elem_match_filter["sentiment"] = sentiment
            # $elemMatch ensures that at least one element in the analysis_results
            # array matches ALL the conditions within elem_match_filter.
            query["analysis_results"] = {"$elemMatch": elem_match_filter}

        try:
            # Fetch documents matching the query.
            # If querying by aspect/sentiment, we fetch the whole document to
            # easily display all analysis results for that review if needed later.
            # Otherwise, fetch only essential fields for a general list.
            if query:
                results = list(self.collection.find(query, {"review_id": 1, "original_text": 1, "stars": 1, "analysis_results": 1}).limit(limit))
            else:
                results = list(self.collection.find({}, {"review_id": 1, "original_text": 1, "stars": 1}).limit(limit))

            # Optional: Post-filter analysis_results in Python if you only want
            # to show the specific matching aspects/sentiments in the display.
            # for doc in results:
            #     if 'analysis_results' in doc and aspect_name:
            #         doc['analysis_results'] = [
            #             res for res in doc['analysis_results']
            #             if res.get('aspect') == aspect_name and (sentiment is None or res.get('sentiment') == sentiment)
            #         ]

            return results
        except Exception as e:
            logger.error(f"Error querying reviews for aspect '{aspect_name}', sentiment '{sentiment}': {e}")
            return [] # Return empty list on error


    def get_distinct_aspects(self) -> List[str]:
        """Gets a list of unique aspect names found in the database."""
        # Corrected check: Use 'is None' for Pymongo collection object
        if self.collection is None:
            logger.error("Cannot get distinct aspects: No MongoDB collection available.")
            return [] # Return empty list

        try:
            # Use the distinct command on the nested field 'analysis_results.aspect'
            distinct_aspects = self.collection.distinct("analysis_results.aspect")
            # Filter out any None or empty string aspects and sort the list
            return sorted([aspect for aspect in distinct_aspects if aspect])
        except Exception as e:
            logger.error(f"Error getting distinct aspects: {e}")
            return [] # Return empty list on error


    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates overall summary statistics."""
        # Corrected check: Use 'is None' for Pymongo collection object
        if self.collection is None:
            logger.error("Cannot get summary stats: No MongoDB collection available.")
            return {"total_reviews": 0, "sentiment_distribution": []} # Return default dict

        try:
            # Get the total count of documents in the collection
            total_reviews = self.collection.count_documents({})

            # Aggregation pipeline to calculate the overall sentiment distribution
            # across all aspects in all documents.
            pipeline = [
                {"$unwind": "$analysis_results"}, # Deconstruct the analysis_results array
                {"$group": {"_id": "$analysis_results.sentiment", "count": {"$sum": 1}}}, # Group by sentiment and count
                {"$project": {"_id": 0, "sentiment": "$_id", "count": 1}}, # Reshape output
                 {"$sort": {"sentiment": 1}} # Sort by sentiment
            ]
            sentiment_distribution = list(self.collection.aggregate(pipeline))

            return {"total_reviews": total_reviews, "sentiment_distribution": sentiment_distribution}
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {"total_reviews": 0, "sentiment_distribution": []} # Return default dict on error