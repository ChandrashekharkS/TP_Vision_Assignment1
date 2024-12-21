import pandas as pd
import ast
import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import os
# Load the dataset
dataset = load_dataset("LDJnr/Puffin")
df = dataset['train'].to_pandas()

# Extract human messages from conversations
def extract_human_messages(conversations):
    if isinstance(conversations, str):
        conversations = ast.literal_eval(conversations)
    return " ".join([msg['value'] for msg in conversations if msg['from'] == 'human'])

df['human_messages'] = df['conversations'].apply(extract_human_messages)

# Generate embeddings using a SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['human_messages'].tolist())

# Perform KMeans clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
df['topic'] = kmeans.fit_predict(embeddings)

# Assign topic labels
topic_labels = {0: "Programming Practices", 1: "Misc"}

# Determine which cluster represents "Programming Practices"
cluster_meanings = {}
for cluster_id in range(2):
    sample_texts = df[df['topic'] == cluster_id]['human_messages'].head(10)
    if any("code" in text.lower() or "programming" in text.lower() or "app" in text.lower() for text in sample_texts):
        cluster_meanings[cluster_id] = "Programming Practices"
    else:
        cluster_meanings[cluster_id] = "Misc"

# Map the identified labels to the clusters
df['topic_label'] = df['topic'].map(cluster_meanings)

# Sentiment Analysis: Use a model with three classes (positive, neutral, negative)
sentiment_analyzer = pipeline("sentiment-analysis", model="bhadresh-savani/bert-base-uncased-emotion")

# Analyze sentiments
def analyze_sentiment(message):
    sentiment_result = sentiment_analyzer(message[:512])[0]
    return sentiment_result['label']

df['sentiment'] = df['human_messages'].apply(analyze_sentiment)

# Save the SentenceTransformer model in its original format
model_save_path = "sentence_transformer_model"
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)

# Save other components with pickle
with open('topic_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('topic_labels.pkl', 'wb') as f:
    pickle.dump(cluster_meanings, f)

print("Model training and saving completed successfully.")
