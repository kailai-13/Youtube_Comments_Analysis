from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from youtube_comment_downloader import *
from itertools import islice

# Load model and tokenizer for Tamil hate speech detection
MODEL_NAME = "Hate-speech-CNERG/tamil-codemixed-abusive-MuRIL"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
CLASS_LABELS = ["Not Abusive", "Abusive"]

def analyze_hate_speech(youtube_link):
    from youtube_comment_downloader import YoutubeCommentDownloader
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    channels = data['channel'].tolist()

    # Load tokenizer and model
    model_name = "Hate-speech-CNERG/tamil-codemixed-abusive-MuRIL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize inputs
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Define class labels
    class_labels = ["Not Abusive", "Abusive"]

    # Extract abusive probabilities and sort them
    abusive_probs = probs[:, 1]
    sorted_indices = torch.argsort(abusive_probs, descending=True)

    # Collect results
    details = []
    categories = {"Not Abusive": 0, "Abusive": 0}

    for idx in sorted_indices:
        text = texts[idx]
        author = authors[idx]
        channel = channels[idx]
        prob = probs[idx]
        pred = torch.argmax(prob).item()
        label = class_labels[pred]
        categories[label] += 1

        details.append({
            'author': author,
            'channel': channel,
            'comment': text,
            'abusive_probability': round(abusive_probs[idx].item() * 100, 2),
            'label': label
        })

    return {
        'categories': list(categories.keys()),
        'counts': list(categories.values()),
        'sorted_comments': details
    }
