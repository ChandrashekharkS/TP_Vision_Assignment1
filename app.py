import streamlit as st
import pandas as pd
import json
import joblib
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import gdown

# Function to download sentence_transformer.joblib from Google Drive
def download_sentence_transformer(file_id, output):
    url = f"https://drive.google.com/file/d/1m3tIItHmGZTrMDewYiZLD73uskQ2zxAJ/view?usp=sharing"
    gdown.download(url, output, quiet=False)

# Load resources
@st.cache_resource
def load_models():
    # Google Drive file ID for sentence_transformer.joblib
    file_id = "1m3tIItHmGZTrMDewYiZLD73uskQ2zxAJ/view"
    sentence_transformer_file = "sentence_transformer.joblib"

    # Check if the file is already downloaded
    if not os.path.exists(sentence_transformer_file):
        st.info("Downloading sentence_transformer.joblib from Google Drive...")
        download_sentence_transformer(file_id, sentence_transformer_file)

    # Load models
    with open('topic_model.joblib', 'rb') as f:
        kmeans = pickle.load(f)

    with open('topic_labels.joblib', 'rb') as f:
        topic_labels = pickle.load(f)

    sentence_transformer = joblib.load(open(sentence_transformer_file, 'rb'))
    return kmeans, topic_labels, sentence_transformer

# Sentiment analysis pipeline with three classes
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="bhadresh-savani/bert-base-uncased-emotion",
)

# Function to extract human messages from JSON structure
def extract_human_messages(conversations):
    messages = []
    for conv in conversations:
        if conv['from'] == 'human':
            messages.append(conv['value'])
    return " ".join(messages)

# Function to truncate messages for sentiment analysis
def truncate_message(message, max_length=512):
    return message[:max_length]

# Sentiment analysis function
def analyze_sentiment(message):
    sentiment_result = sentiment_analyzer(message[:512])[0]
    return sentiment_result['label']

# Main data processing function
def process_data(file, kmeans, topic_labels, sentence_transformer):
    # Load JSON file
    data = json.load(file)
    df = pd.DataFrame(data)

    # Extract human messages
    df['human_messages'] = df['conversations'].apply(extract_human_messages)
    df['human_messages'] = df['human_messages'].apply(lambda x: truncate_message(x, 512))

    # Generate embeddings for clustering
    embeddings = sentence_transformer.encode(df['human_messages'].tolist())

    # Predict topics
    df['topic'] = kmeans.predict(embeddings)
    df['topic_label'] = df['topic'].map(topic_labels)

    # Handle edge case: assign "Misc" if topic label is missing
    df['topic_label'].fillna("Misc", inplace=True)

    # Analyze sentiments
    df['sentiment'] = df['human_messages'].apply(analyze_sentiment)

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
    total_pages = (len(df) // page_size) + 1
    page_number = st.number_input("Page Number", min_value=1, max_value=total_pages, step=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    paginated_df = df.iloc[start_idx:end_idx]
    st.table(paginated_df[['human_messages', 'topic_label', 'sentiment']])

# Streamlit App
st.title("Conversation Topic and Sentiment Analysis")

# Load all models
kmeans, topic_labels, sentence_transformer = load_models()

# File uploader
uploaded_file = st.file_uploader("Upload a JSON file containing conversations", type="json")

if uploaded_file is not None:
    df, topic_counts, sentiment_counts = process_data(uploaded_file, kmeans, topic_labels, sentence_transformer)

    # Navigation sidebar
    page = st.sidebar.selectbox("Select Page", ["Counts", "Sessions"])

    if page == "Counts":
        display_counts(topic_counts, sentiment_counts)
    elif page == "Sessions":
        display_sessions(df)
