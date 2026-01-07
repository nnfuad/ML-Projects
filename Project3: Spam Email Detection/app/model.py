from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_model():
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.95
    )

    model = MultinomialNB()
    return model, vectorizer