import streamlit as st
import pandas as pd
import json
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Load pre-trained models
@st.cache_resource
def load_models():
    # Load the SentenceTransformer model
    sentence_transformer = SentenceTransformer("sentence_transformer_model")

    # Load the KMeans topic model
    with open("topic_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    # Load topic labels
    with open("topic_labels.pkl", "rb") as f:
        topic_labels = pickle.load(f)

    # Load sentiment analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="bhadresh-savani/bert-base-uncased-emotion"
    )
    
    return sentence_transformer, kmeans, topic_labels, sentiment_analyzer

# Function to extract human messages from JSON structure
def extract_human_messages(conversations):
    messages = []
    for conv in conversations:
        if conv['from'] == 'human':
            messages.append(conv['value'])
    return " ".join(messages)

# Sentiment analysis function
def analyze_sentiment(sentiment_analyzer, message, max_length=512):
    truncated_message = message[:max_length]
    sentiment_result = sentiment_analyzer(truncated_message)[0]
    return sentiment_result['label']

# Main data processing function
def process_data(file, sentence_transformer, kmeans, topic_labels, sentiment_analyzer):
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

# Function to display aggregated counts
def display_counts(topic_counts, sentiment_counts):
    st.header("Counts")
    st.subheader("Table 1: Topic Counts")
    st.table(topic_counts)

    st.subheader("Table 2: Sentiment Counts")
    st.table(sentiment_counts)

# Function to display session details with pagination
def display_sessions(df):
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
    df, topic_counts, sentiment_counts = process_data(
        uploaded_file, 
        sentence_transformer, 
        kmeans, 
        topic_labels, 
        sentiment_analyzer
    )

    # Navigation sidebar
    page = st.sidebar.selectbox("Select Page", ["Counts", "Sessions"])

    if page == "Counts":
        display_counts(topic_counts, sentiment_counts)
    elif page == "Sessions":
        display_sessions(df)
