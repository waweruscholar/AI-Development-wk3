import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I absolutely love the Apple AirPods! The sound quality is amazing.",
    "The Samsung Galaxy S21 is overpriced and the battery life is terrible.",
    "This Logitech keyboard works like a charm and feels great to type on.",
    "I'm disappointed with the Sony headphones. Not worth the price.",
    "Bought a Dell XPS laptop and it's blazing fast and reliable!"
]

# Rule-based sentiment keywords
positive_keywords = ['love', 'great', 'amazing', 'charm', 'fast', 'reliable']
negative_keywords = ['disappointed', 'terrible', 'overpriced', 'not worth']

# Process and analyze each review
for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    
    # Named Entity Recognition
    print("Entities (PRODUCT/ORG):")
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG"):
            print(f" - {ent.text} ({ent.label_})")
    
    # Rule-based Sentiment
    review_lower = review.lower()
    sentiment = "Neutral"
    if any(word in review_lower for word in positive_keywords):
        sentiment = "Positive"
    elif any(word in review_lower for word in negative_keywords):
        sentiment = "Negative"
    
    print(f"Sentiment: {sentiment}")
