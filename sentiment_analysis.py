import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download required NLTK data (run once)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze the sentiment of input text and return results
    """
    # Get sentiment scores
    scores = sid.polarity_scores(text)

    # Determine sentiment based on compound score
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return {
        'sentiment': sentiment,
        'scores': {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    }

def main():
    # Example texts to analyze
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible, worst experience ever. Completely disappointed.",
        "The weather is okay today, nothing special."
    ]

    # Create a list to store results
    results = []

    # Analyze each text
    for text in sample_texts:
        result = analyze_sentiment(text)
        results.append({
            'Text': text,
            'Sentiment': result['sentiment'],
            'Compound Score': result['scores']['compound'],
            'Positive': result['scores']['positive'],
            'Negative': result['scores']['negative'],
            'Neutral': result['scores']['neutral']
        })

    # Create a DataFrame for better visualization
    df = pd.DataFrame(results)

    # Print results
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    print(df.to_string(index=False))

    # Optional: Save to CSV
    df.to_csv('sentiment_analysis_results.csv', index=False)
    print("\nResults saved to 'sentiment_analysis_results.csv'")

if __name__ == "__main__":
    # Example of analyzing user input
    print("Enter text for sentiment analysis (or 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        result = analyze_sentiment(user_input)
        print(f"\nSentiment: {result['sentiment']}")
        print(f"Scores: {result['scores']}\n")

    # Run the main analysis with sample texts
    main()
