
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Step 1: Load the dataset
df = pd.read_csv('C:\\Users\\kavit\\AppData\\Local\\Temp\\26fd8d62-7b25-4046-9d71-1344afeeabca_archive (3).zip.bca\\twitter_validation')

# Step 2: Text preprocessing (e.g., remove URLs, mentions, special characters, etc.)

# Step 3: Sentiment Analysis
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Step 4: Sentiment Visualization
sentiment_distribution = df['sentiment'].value_counts()
sentiment_distribution.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Step 6: Word Clouds for each sentiment category
from wordcloud import WordCloud

positive_words = ' '.join(df[df['sentiment'] == 'positive']['text'])
negative_words = ' '.join(df[df['sentiment'] == 'negative']['text'])
neutral_words = ' '.join(df[df['sentiment'] == 'neutral']['text'])

# Generate word clouds
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_words)
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_words)

# Plot word clouds
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Sentiment')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Sentiment')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.title('Neutral Sentiment')
plt.axis('off')

plt.show()
