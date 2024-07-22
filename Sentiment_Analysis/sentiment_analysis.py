import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load data
data = pd.read_csv('social_media_posts.csv')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiments
data['sentiment'] = data['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Save results
data.to_csv('sentiment_results.csv', index=False)
