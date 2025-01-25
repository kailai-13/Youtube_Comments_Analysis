from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from youtube_comment_downloader import *
from itertools import islice

# Load the advanced emotion classification model and tokenizer globally for efficiency
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"  # Advanced emotion classification model
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Emotion labels for the advanced model
EMOTION_LABELS = ["anger", "joy", "sadness", "neutral", "fear", "surprise", "disgust"]


def analyze_sentiment(youtube_link):
    # Download comments
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(youtube_link, sort_by=SORT_BY_POPULAR)
    all_comments = [comment for comment in islice(comments, 2000)]
    df = pd.DataFrame(all_comments)

    # Preprocess the data
    data = df.drop(['cid', 'time', 'votes', 'replies', 'photo', 'heart', 'reply', 'time_parsed'], axis=1)
    data.columns = ['comment', 'author', 'channel']
    texts = data['comment'].tolist()
    authors = data['author'].tolist()

    # Tokenize inputs
    inputs = TOKENIZER(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Perform inference
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Get predicted emotions
    results = []
    categories = {label: 0 for label in EMOTION_LABELS}

    for idx, text in enumerate(texts):
        prob = probs[idx]
        pred = torch.argmax(prob).item()
        emotion = EMOTION_LABELS[pred]
        categories[emotion] += 1

        results.append({
            'author': authors[idx],
            'comment': text,
            'emotion': emotion
        })

    return {
        'categories': list(categories.keys()),
        'counts': list(categories.values()),
        'comments_with_emotions': results
    }
