import pandas as pd
import re
from transformers import pipeline
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script
data_dir = os.path.join(current_dir, "..", "data")        # ../data relative to notebook
os.makedirs(data_dir, exist_ok=True)  # <-- create 'data' folder if it doesn't exist

data_file = os.path.join(data_dir, "sample_transcripts.txt")

# Load transcript
with open(data_file, 'r') as f:
    text = f.read()

# Preprocess: extract speaker and sentence
lines = text.split('\n')
data = []
for line in lines:
    match = re.match(r"(Speaker \d+): (.+)", line)
    if match:
        speaker, sentence = match.groups()
        data.append({"speaker": speaker, "sentence": sentence})

df = pd.DataFrame(data)

# Save structured CSV
structured_csv = os.path.join(data_dir, "structured_transcripts.csv")
df.to_csv(structured_csv, index=False)

# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
df['sentiment'] = df['sentence'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

# Save final CSV with sentiment
final_csv = os.path.join(data_dir, "structured_transcripts_with_sentiment.csv")
df.to_csv(final_csv, index=False)

print("CSV files generated successfully in 'data/' folder!")
