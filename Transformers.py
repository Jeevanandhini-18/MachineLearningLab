from transformers import pipeline

# Load pretrained sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

texts = [
    "The service was slow but staff were polite",
    "Customer support did not solve my issue",
    "Booking process was simple and quick",
    "Waiting time was longer than expected",
    "Overall experience was satisfactory",
    "The website is confusing to use",
    "Response time was reasonable",
    "Not happy with the refund process",
    "The staff explained everything clearly",
    "The process needs improvement"
]

for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"Text: {text}")
    print(f"Prediction: {result['label']} | Score: {result['score']:.2f}")
    print("-" * 50)