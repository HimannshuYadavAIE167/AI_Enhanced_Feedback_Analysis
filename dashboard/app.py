# standalone_dashboard_folder/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust 'os.pardir' based on how many levels up the project root is
# If config and database are in the same folder as app.py, remove these sys.path lines
# project_root = os.path.abspath(os.path.join(current_dir, os.pardir)) # Example if one level up
# sys.path.insert(0, project_root)


# Project modules (now importable because config.py and database/ are in the same folder)
try:
    # Ensure config.py is in the same directory
    import config
    # Ensure database/mongo_client.py is at database/mongo_client.py relative to this script
    from database.mongo_client import MongoClientWrapper
except ImportError as e:
    st.error(f"Error importing project modules. Make sure config.py and the 'database' folder are in the same directory as this script. Details: {e}")
    st.stop() # Stop the app if core modules cannot be imported


# Setup logging (optional, Streamlit has its own logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Database Connection ---
# Use Streamlit's caching to initialize the client only once across reruns
@st.cache_resource
def get_mongo_client():
    """Initializes and returns the MongoDB client wrapper."""
    try:
        # Use details from the imported config
        client = MongoClientWrapper(
            uri=config.MONGO_URI,
            db_name=config.MONGO_DB_NAME,
            collection_name=config.MONGO_COLLECTION_NAME
        )
        # Note: create_indexes is typically done by the batch inference script.
        # Running it here on every dashboard start might be slow on large collections.
        # Uncomment only if you are sure it's needed and won't impact performance.
        # client.create_indexes()
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        logger.error(f"Dashboard MongoDB connection failed: {e}")
        return None

# --- Data Fetching Functions (Cached) ---
# Use Streamlit's data caching to avoid re-fetching data unnecessarily
@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_distinct_aspects(_client):
    """Fetches distinct aspects from the database."""
    if _client is not None: # Use explicit None check
        logger.info("Fetching distinct aspects from DB.")
        return _client.get_distinct_aspects()
    return [] # Return empty list if client is None

@st.cache_data(ttl=600)
def fetch_summary_stats(_client):
    """Fetches summary statistics."""
    if _client is not None: # Use explicit None check
        logger.info("Fetching summary stats from DB.")
        return _client.get_summary_stats()
    # Return default structure if client is None
    return {"total_reviews": 0, "sentiment_distribution": []}

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_sentiments_for_aspect(_client, aspect_name):
    """Fetches sentiment distribution for a specific aspect."""
    # Use explicit None check for client
    if _client is not None and aspect_name and aspect_name != "All":
        logger.info(f"Fetching sentiment distribution for aspect: {aspect_name}")
        return _client.get_sentiments_by_aspect(aspect_name)
    elif aspect_name == "All" and _client is not None: # Use explicit None check
         # If "All" is selected, fetch overall distribution
         summary = fetch_summary_stats(_client)
         return summary.get("sentiment_distribution", [])
    return [] # Return empty list if client is None or aspect is invalid

@st.cache_data(ttl=300)
def fetch_reviews(_client, aspect_name=None, sentiment=None, limit=10):
    """Fetches sample reviews based on filters."""
    if _client is not None: # Use explicit None check
        logger.info(f"Fetching sample reviews (Aspect: {aspect_name}, Sentiment: {sentiment}, Limit: {limit})")
        # Pass None if "All" is selected for aspect or sentiment filter
        query_aspect = aspect_name if aspect_name != "All" else None
        query_sentiment = sentiment if sentiment != "All" else None
        return _client.get_reviews_by_sentiment(query_aspect, query_sentiment, limit)
    return [] # Return empty list if client is None


# --- Streamlit App Layout ---

# Page Configuration (do this first)
st.set_page_config(
    page_title="Yelp ABSA Dashboard",
    page_icon="📊",
    layout="wide", # Use wide layout for dashboards
    initial_sidebar_state="expanded"
)

# Initialize MongoDB client (this will run the cached function)
client = get_mongo_client()

# --- Sidebar ---
st.sidebar.title("📊 Yelp ABSA Explorer")
st.sidebar.markdown("Analyze customer feedback with Aspect-Based Sentiment Analysis.")

# Check if client connection was successful
if client is not None: # Use explicit None check
    # Fetch distinct aspects for the dropdown
    distinct_aspects = fetch_distinct_aspects(client)
    # Add "All" option and fallback to config aspects if DB is empty
    # Use getattr to safely access PREDEFINED_ASPECTS from config
    available_aspects = ["All"] + (distinct_aspects if distinct_aspects else getattr(config, 'PREDEFINED_ASPECTS', []))

    # Aspect Selection
    selected_aspect = st.sidebar.selectbox(
        "Select Aspect to Analyze:",
        options=available_aspects,
        index=0 # Default to "All"
    )

    # Sentiment Filter (Optional)
    sentiment_options = ["All", "Positive", "Negative", "Neutral"]
    selected_sentiment_filter = st.sidebar.radio(
        "Filter by Sentiment:",
        options=sentiment_options,
        index=0 # Default to "All"
    )

    # Limit for displayed reviews
    review_limit = st.sidebar.slider("Max Reviews to Display:", min_value=5, max_value=50, value=10, step=5)

else:
    # Display message if database connection failed
    st.sidebar.error("Database connection failed. Cannot load filters.")
    selected_aspect = "All" # Default values if no DB connection
    selected_sentiment_filter = "All"
    review_limit = 10
    # Provide dummy options if no DB connection
    available_aspects = ["All"] + getattr(config, 'PREDEFINED_ASPECTS', [])


# --- Main Area ---
st.title("AI-Enhanced Customer Feedback Analysis")
st.markdown("Insights from Yelp Restaurant Reviews")

# Display content only if client is connected
if client is not None: # Use explicit None check
    # --- Overview Section ---
    st.header("Overall Summary")
    summary_stats = fetch_summary_stats(client)
    col1, col2 = st.columns(2) # Create two columns for layout
    with col1:
        st.metric("Total Reviews Analyzed", summary_stats.get("total_reviews", 0))

    with col2:
        st.subheader("Overall Sentiment Distribution (All Aspects)")
        overall_sentiment_data = summary_stats.get("sentiment_distribution", [])

        # --- FIX: Ensure all sentiment categories are present in the DataFrame ---
        # Create a DataFrame with all expected sentiments and merge with fetched data
        all_sentiments_df = pd.DataFrame({'sentiment': ["Positive", "Neutral", "Negative"], 'count': 0})
        overall_sentiment_df = pd.DataFrame(overall_sentiment_data)

        if not overall_sentiment_df.empty:
             # Merge fetched data into the all_sentiments_df, summing counts
             overall_sentiment_df = pd.merge(all_sentiments_df, overall_sentiment_df, on='sentiment', how='left', suffixes=('_all', '_fetched'))
             overall_sentiment_df['count'] = overall_sentiment_df['count_all'].fillna(0) + overall_sentiment_df['count_fetched'].fillna(0)
             overall_sentiment_df = overall_sentiment_df[['sentiment', 'count']] # Keep only necessary columns
             # Ensure sentiment column is categorical for consistent ordering/colors
             overall_sentiment_df['sentiment'] = pd.Categorical(overall_sentiment_df['sentiment'], categories=["Positive", "Neutral", "Negative"], ordered=True)
             overall_sentiment_df = overall_sentiment_df.sort_values('sentiment')
        else:
             # If no data fetched, use the all_sentiments_df with counts of 0
             overall_sentiment_df = all_sentiments_df
             overall_sentiment_df['sentiment'] = pd.Categorical(overall_sentiment_df['sentiment'], categories=["Positive", "Neutral", "Negative"], ordered=True)
             overall_sentiment_df = overall_sentiment_df.sort_values('sentiment')


        # Only plot if there's data (even if counts are 0, the rows exist)
        if not overall_sentiment_df.empty:
            fig_overall = px.pie(overall_sentiment_df, names='sentiment', values='count',
                                 title="Overall Sentiment Distribution (All Aspects)",
                                 color='sentiment', # Use sentiment for color mapping
                                 color_discrete_map={'Positive':'green', 'Neutral':'grey', 'Negative':'red'}, # Define colors
                                 # Optional: textinfo='percent+value' to show percentage and count
                                 # hole=0.4 # Optional: make it a donut chart
                                 )
            st.plotly_chart(fig_overall, use_container_width=True)
        else:
            st.write("No overall sentiment data available.")
        # --- End FIX ---


    st.divider() # Add a visual separator

    # --- Aspect-Specific Section ---
    # Only show aspect-specific details if a specific aspect is selected
    if selected_aspect != "All":
        st.header(f"Analysis for Aspect: '{selected_aspect}'")

        st.subheader(f"Sentiment Distribution for '{selected_aspect}'")
        # Sentiment distribution for the selected aspect
        aspect_sentiments = fetch_sentiments_for_aspect(client, selected_aspect)
        aspect_sentiment_df = pd.DataFrame(aspect_sentiments)

        # --- FIX: Ensure all sentiment categories are present for aspect-specific chart ---
        # Create a DataFrame with all expected sentiments and merge with fetched data
        aspect_all_sentiments_df = pd.DataFrame({'sentiment': ["Positive", "Neutral", "Negative"], 'count': 0})

        if not aspect_sentiment_df.empty:
             # Merge fetched data into the aspect_all_sentiments_df, summing counts
             aspect_sentiment_df = pd.merge(aspect_all_sentiments_df, aspect_sentiment_df, on='sentiment', how='left', suffixes=('_all', '_fetched'))
             aspect_sentiment_df['count'] = aspect_sentiment_df['count_all'].fillna(0) + aspect_sentiment_df['count_fetched'].fillna(0)
             aspect_sentiment_df = aspect_sentiment_df[['sentiment', 'count']] # Keep only necessary columns
             # Ensure sentiment column is categorical for consistent ordering/colors
             aspect_sentiment_df['sentiment'] = pd.Categorical(aspect_sentiment_df['sentiment'], categories=["Positive", "Neutral", "Negative"], ordered=True)
             aspect_sentiment_df = aspect_sentiment_df.sort_values('sentiment')
        else:
             # If no data fetched, use the aspect_all_sentiments_df with counts of 0
             aspect_sentiment_df = aspect_all_sentiments_df
             aspect_sentiment_df['sentiment'] = pd.Categorical(aspect_sentiment_df['sentiment'], categories=["Positive", "Neutral", "Negative"], ordered=True)
             aspect_sentiment_df = aspect_sentiment_df.sort_values('sentiment')
        # --- End FIX ---


        if not aspect_sentiment_df.empty:
            fig_aspect = px.bar(aspect_sentiment_df, x='sentiment', y='count',
                                 title=f"Sentiment Distribution for '{selected_aspect}'",
                                 labels={'count': 'Number of Mentions', 'sentiment': 'Sentiment'},
                                 color='sentiment', # Use sentiment for color mapping
                                 color_discrete_map={'Positive':'green', 'Neutral':'grey', 'Negative':'red'}) # Define colors
            st.plotly_chart(fig_aspect, use_container_width=True)
        else:
            st.write(f"No sentiment data found for aspect '{selected_aspect}' in the database.")

        st.subheader("Sample Reviews")
        # Determine the sentiment filter to apply
        filter_sentiment = selected_sentiment_filter if selected_sentiment_filter != "All" else None
        # Fetch sample reviews based on the selected aspect and sentiment filter
        sample_reviews = fetch_reviews(client, selected_aspect, filter_sentiment, review_limit)

        if sample_reviews:
            # Display each sample review
            for review in sample_reviews:
                st.markdown(f"**Review ID:** `{review.get('review_id')}` | **Stars:** {review.get('stars', 'N/A')}")

                # Find and display the specific analysis results for the selected aspect/sentiment within this review
                relevant_analysis = []
                if 'analysis_results' in review and review['analysis_results']:
                    for res in review['analysis_results']:
                        is_aspect_match = res.get('aspect') == selected_aspect
                        is_sentiment_match = filter_sentiment is None or res.get('sentiment') == filter_sentiment
                        if is_aspect_match and is_sentiment_match:
                             # Format the relevant analysis result
                             relevant_analysis.append(f" -> Aspect: **{res.get('aspect')}**, Sentiment: **{res.get('sentiment')}** (Conf: {res.get('confidence', 0.0):.2f})")

                # Use an expander to show the full review text
                with st.expander("View Full Review Text"):
                    st.write(review.get('original_text', 'N/A'))
                    # Display the relevant analysis results if found
                    if relevant_analysis:
                         st.markdown("Relevant Mentions:")
                         for item in relevant_analysis:
                             st.markdown(item)
                st.markdown("---") # Separator between reviews
        else:
            st.write(f"No reviews found matching the criteria (Aspect: {selected_aspect}, Sentiment: {selected_sentiment_filter}).")

    else: # Case where "All" aspects is selected
        st.header("Recent Reviews")
        # Fetch recent reviews without filtering by a specific aspect or sentiment
        sample_reviews = fetch_reviews(client, limit=review_limit)
        if sample_reviews:
            # Display each recent review
            for review in sample_reviews:
                st.markdown(f"**Review ID:** `{review.get('review_id')}` | **Stars:** {review.get('stars', 'N/A')}")
                with st.expander("View Full Review Text"):
                    st.write(review.get('original_text', 'N/A'))
                    # Optionally display all analysis results for this review
                    if 'analysis_results' in review and review['analysis_results']:
                         st.markdown("All Analysis Results:")
                         for res in review['analysis_results']:
                             st.markdown(f" -> Aspect: {res.get('aspect')}, Sentiment: {res.get('sentiment')} (Conf: {res.get('confidence', 0.0):.2f})")
                st.markdown("---") # Separator between reviews
        else:
            st.write("No reviews found in the database.")

else:
    # Message displayed if the initial database connection failed
    st.error("Application could not connect to the database. Please check your MongoDB server status and the connection URI in config.py.")

# Add a footer or info section
st.sidebar.markdown("---")
st.sidebar.info("This dashboard visualizes Aspect-Based Sentiment Analysis results.")

# Add a note about data source if desired
# st.sidebar.caption("Data Source: Yelp Academic Dataset (Subset)")

