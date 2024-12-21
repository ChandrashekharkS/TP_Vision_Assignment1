import streamlit as st
import pandas as pd
import json
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import gdown

# Google Drive folder ID
folder_id = "1iaox1qYkvhu1biGmgVXb6yKADdFPBXjD"
folder_path = "sentence_transformer_model"

# Construct the Google Drive download folder URL
folder_url = f"https://drive.google.com/drive/folders/1iaox1qYkvhu1biGmgVXb6yKADdFPBXjD?usp=sharing"

# Check if the folder already exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    print("Downloading the folder from Google Drive...")
    # Download the entire folder
    gdown.download_folder(folder_url, quiet=False)
    print("Download completed!")
else:
    print("Folder already exists.")

# Load pre-trained models
@st.cache_resource
def load_models():
    """
    Loads the pre-trained models for sentence transformers, KMeans clustering,
    topic labels, and sentiment analysis pipeline.
    """
    try:
        # Load the SentenceTransformer model from the local folder
        sentence_transformer = SentenceTransformer(folder_path)

        # Load the KMeans topic model
        with open(f"{folder_path}/topic_model.pkl", "rb") as f:
            kmeans = pickle.load(f)

        # Load topic labels
        with open(f"{folder_path}/topic_labels.pkl", "rb") as f:
            topic_labels = pickle.load(f)

        # Load sentiment analysis pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="bhadresh-savani/bert-base-uncased-emotion"
        )
        
        return sentence_transformer, kmeans, topic_labels, sentiment_analyzer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Function to extract human messages from JSON structure
def extract_human_messages(conversations):
    """
    Extracts human messages from the conversation JSON structure.
    """
    messages = []
    for conv in conversations:
        if conv['from'] == 'human':
            messages.append(conv['value'])
    return " ".join(messages)

# Sentiment analysis function
def analyze_sentiment(sentiment_analyzer, message, max_length=512):
    """
    Analyzes sentiment of a given message using the sentiment analyzer.
    """
    truncated_message = message[:max_length]
    sentiment_result = sentiment_analyzer(truncated_message)[0]
    return sentiment_result['label']

# Main data processing function
def process_data(file, sentence_transformer, kmeans, topic_labels, sentiment_analyzer):
    """
    Processes the uploaded JSON file and performs topic and sentiment analysis.
    """
    try:
        # Load JSON file
        data = json.load(file)
        df = pd.DataFrame(data)

        # Extract human messages
        df['human_messages'] = df['conversations'].apply(extract_human_messages)

        # Generate embeddings for clustering
        embeddings = sentence_transformer.encode(df['human_messages'].tolist())

        # Predict topics
        df['topic'] = kmeans.predict(embeddings)
        df['topic_label'] = df['topic'].map(topic_labels)

        # Handle edge case: assign "Misc" if topic label is missing
        df['topic_label'].fillna("Misc", inplace=True)

        # Analyze sentiments
        df['sentiment'] = df['human_messages'].apply(lambda x: analyze_sentiment(sentiment_analyzer, x))

        # Aggregate counts for display
        topic_counts = df['topic_label'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']

        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        return df, topic_counts, sentiment_counts
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None

# Function to display aggregated counts
def display_counts(topic_counts, sentiment_counts):
    """
    Displays the aggregated counts of topics and sentiments in table format.
    """
    st.header("Counts")
    st.subheader("Table 1: Topic Counts")
    st.table(topic_counts)

    st.subheader("Table 2: Sentiment Counts")
    st.table(sentiment_counts)

# Function to display session details with pagination
def display_sessions(df):
    """
    Displays paginated session details, including human messages, topic labels, and sentiments.
    """
    st.header("Sessions")
    st.subheader("Assigned Topics and Sentiments")
    st.write("Paginated view of conversations:")

    page_size = 50
    page_number = st.number_input("Page Number", min_value=1, max_value=(len(df) // page_size) + 1, step=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    paginated_df = df.iloc[start_idx:end_idx]
    st.table(paginated_df[['human_messages', 'topic_label', 'sentiment']])

# Streamlit App
st.title("Conversation Topic and Sentiment Analysis")

# Load models
sentence_transformer, kmeans, topic_labels, sentiment_analyzer = load_models()

# File uploader
uploaded_file = st.file_uploader("Upload a JSON file containing conversations", type="json")

if uploaded_file is not None:
    # Process the data and handle errors
    df, topic_counts, sentiment_counts = process_data(
        uploaded_file, 
        sentence_transformer, 
        kmeans, 
        topic_labels, 
        sentiment_analyzer
    )

    if df is not None and topic_counts is not None and sentiment_counts is not None:
        # Navigation sidebar
        page = st.sidebar.selectbox("Select Page", ["Counts", "Sessions"])

        if page == "Counts":
            display_counts(topic_counts, sentiment_counts)
        elif page == "Sessions":
            display_sessions(df)
