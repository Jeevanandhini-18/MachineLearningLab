from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Dataset
texts = [
    "The phone works very well",
    "Battery drains too fast",
    "Camera quality is excellent",
    "Screen stopped working",
    "Fast performance and smooth usage",
    "Device heats up quickly",
    "Sound quality is clear",
    "Charging takes too long",
    "Good build and design",
    "Phone crashes frequently"
]

labels = [
    "Positive", "Negative",
    "Positive", "Negative",
    "Positive", "Negative",
    "Positive", "Negative",
    "Positive", "Negative"
]

# Step 1: Convert text to numbers (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
X = X.toarray()

# Step 2: Prior probabilities
classes, counts = np.unique(labels, return_counts=True)
priors = {c: count / len(labels) for c, count in zip(classes, counts)}

print("Prior Probabilities:")
for c in priors:
    print(f"P({c}) = {priors[c]:.2f}")

# Step 3: Likelihoods (Laplace smoothing)
likelihoods = {}
V = len(vocab)

for c in classes:
    X_c = X[np.array(labels) == c]
    word_counts = X_c.sum(axis=0) + 1
    total_words = word_counts.sum()
    likelihoods[c] = word_counts / total_words

# Step 4: Prediction
print("\nPosterior Probabilities and Predictions:")

for i, text in enumerate(texts, 1):
    x_new = vectorizer.transform([text]).toarray()[0]
    posteriors = {}

    for c in classes:
        posterior = priors[c]
        for idx, count in enumerate(x_new):
            if count > 0:
                posterior *= likelihoods[c][idx] ** count
        posteriors[c] = posterior

    predicted = max(posteriors, key=posteriors.get)

    print(f"\nID {i}: {text}")
    for c in posteriors:
        print(f"  P({c} | sentence) = {posteriors[c]:.8f}")
    print(f"  ➜ Predicted Sentiment: {predicted}")