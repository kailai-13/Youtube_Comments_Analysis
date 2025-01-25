from django.shortcuts import render
from .sentiment_model import analyze_sentiment
from .hate_speech_model import analyze_hate_speech
import matplotlib.pyplot as plt
import io
import base64

# Home Page View
def home(request):
    return render(request, 'home.html')

# Sentiment Analysis View
def sentiment_analysis(request):
    result = None
    charts = None
    comments_with_emotions = []

    if request.method == "POST":
        youtube_link = request.POST.get('youtube_link')
        result = analyze_sentiment(youtube_link)

        # Generate bar chart
        fig, ax = plt.subplots()
        categories = result['categories']
        counts = result['counts']
        ax.bar(categories, counts, color=['blue', 'green', 'red', 'gray', 'purple', 'orange', 'brown'])
        ax.set_title('Sentiment Analysis Results')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Emotions')

        # Save chart as an image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        charts = f"data:image/png;base64,{base64.b64encode(image_png).decode()}"

        # Get comments with emotions for display
        comments_with_emotions = result['comments_with_emotions']

    return render(
        request,
        'sentiment_analysis.html',
        {
            'result': result,
            'charts': charts,
            'comments_with_emotions': comments_with_emotions
        }
    )


# Hate Speech Detection View
def hate_speech_analysis(request):
    result = None
    charts = None
    sorted_abusive_comments = []

    if request.method == "POST":
        youtube_link = request.POST.get('youtube_link')
        result = analyze_hate_speech(youtube_link)

        # Generate bar chart
        fig, ax = plt.subplots()
        categories = result['categories']
        counts = result['counts']
        ax.bar(categories, counts, color=['blue', 'red'])
        ax.set_title('Hate Speech Analysis Results')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Hate Speech Categories')

        # Save chart as an image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        charts = f"data:image/png;base64,{base64.b64encode(image_png).decode()}"

        # Get sorted comments for display
        sorted_abusive_comments = result['sorted_comments']

    return render(
        request,
        'hate_speech_analysis.html',
        {
            'result': result,
            'charts': charts,
            'sorted_abusive_comments': sorted_abusive_comments
        }
    )
