from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(text):

    result = sentiment_pipeline(text)[0]

    sentiment = result["label"]
    confidence = round(result["score"], 2)

    return f"{sentiment} (confidence {confidence})"