import pandas as pd
import re
import os

# Check if the CSV file exists
csv_file = r'c:\Users\HP\Downloads\flipkart_data (1).csv'
if not os.path.exists(csv_file):
    print(f"❌ Error: {csv_file} not found.")
    print("Please ensure the Flipkart dataset CSV file is present at the specified path.")
    print("Expected columns: 'review' and 'rating'")
    exit(1)

# Load original dataset (your attached file)
df = pd.read_csv(csv_file)

# Clean reviews: remove 'READ MORE' and extra spaces
df['review_clean'] = df['review'].str.replace('READ MORE', '', regex=False).str.strip()

# Map ratings to binary sentiment: 4-5=1 (positive), 1-3=0 (negative)
df['sentiment'] = (df['rating'] >= 4).astype(int)

# Remove very short reviews (<10 chars)
df_filtered = df[df['review_clean'].str.len() > 10].copy()

# Text preprocessing: lowercase + alphanumeric only
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text.strip()

df_filtered['text'] = df_filtered['review_clean'].apply(preprocess)

# Final improved dataset (text, sentiment)
improved_df = df_filtered[['text', 'sentiment']].dropna()

# Save to CSV
improved_df.to_csv('improved_flipkart.csv', index=False)
print(f"✅ Saved {len(improved_df)} rows to improved_flipkart.csv")
print("\nDataset preview:")
print(improved_df.head())
print("\nLabel distribution:")
print(improved_df['sentiment'].value_counts())
