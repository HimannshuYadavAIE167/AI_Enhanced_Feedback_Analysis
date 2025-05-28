# main.py (Optional Orchestrator)
import argparse
import logging
import sys

# Add project root to Python path to allow imports from subdirectories
# This might be necessary depending on how you run the script
# import os
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from absa_model.batch_inference import run_batch_inference
# Add imports for other potential steps like data downloading/preprocessing if separated

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI-Enhanced Customer Feedback Analysis Pipeline Orchestrator.")
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run the batch inference process (load, preprocess, predict, store)."
    )
    # Add other arguments for different pipeline stages if needed
    # parser.add_argument("--preprocess-only", action="store_true", help="Run only data preprocessing.")
    # parser.add_argument("--setup-db", action="store_true", help="Initialize database and create indexes.")

    args = parser.parse_args()

    if args.run_inference:
        logger.info("Starting the batch inference pipeline...")
        try:
            run_batch_inference()
            logger.info("Batch inference pipeline finished successfully.")
        except Exception as e:
            logger.error(f"Batch inference pipeline failed: {e}", exc_info=True)
            sys.exit(1)
    # Add elif blocks for other arguments
    # elif args.setup_db:
    #     logger.info("Setting up database...")
    #     # Call DB setup function
    else:
        logger.info("No action specified. Use --help to see available options.")
        parser.print_help()

if __name__ == "__main__":
    main()