import streamlit as st
import pandas as pd
import os

# Make sure the path works no matter where you run it from
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "structured_transcripts_with_sentiment.csv")

# Load processed data
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"File not found at {data_path}. Please run the ETL notebook first to generate the CSV.")
    st.stop()

st.title("Meeting Insights Dashboard 🚀")

# Speaker-level sentiment summary
speaker_summary = df.groupby(['speaker', 'sentiment']).size().unstack(fill_value=0)
st.subheader("Speaker-level Sentiment")
st.bar_chart(speaker_summary)

# Display transcript with sentiment
st.subheader("Transcript with Sentiment")
st.dataframe(df)
