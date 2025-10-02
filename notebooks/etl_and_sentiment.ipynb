import pandas as pd
import re
from transformers import pipeline

# Load transcript
with open('data/sample_transcripts.txt', 'r') as f:
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
df.to_csv("data/structured_transcripts.csv", index=False)

# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
df['sentiment'] = df['sentence'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
df.to_csv("data/structured_transcripts_with_sentiment.csv", index=False)

print("CSV generated successfully!")
