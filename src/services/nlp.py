from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import re

_analyzer = SentimentIntensityAnalyzer()

def sentiment_scores(text: str) -> Dict[str, float]:
    s = _analyzer.polarity_scores(text or "")
    # Map to positive/neutral/negative proportions
    return {"positive": round(max(s["pos"], 0.0), 4),
            "neutral": round(max(s["neu"], 0.0), 4),
            "negative": round(max(s["neg"], 0.0), 4)}

def readability_score(text: str) -> float:
    try:
        return float(textstat.flesch_reading_ease(text or ""))
    except Exception:
        return 60.0

def clean_tokens(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    return words

def extract_topics_and_keywords(texts: List[str], top_k: int = 8) -> Dict[str, List[str]]:
    """Use TF-IDF to extract top corpus terms as topics/keywords."""
    if not texts:
        return {"topics": [], "keywords": []}
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    terms_only = [t for t, _ in ranked[: max(5, top_k)]]
    return {"topics": terms_only[: top_k], "keywords": terms_only[: top_k]}

def tfidf_similarity(a: str, b: str) -> float:
    if not a.strip() and not b.strip():
        return 0.0  # nothing to compare
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer()
    try:
        X = vec.fit_transform([a or "", b or ""])
    except ValueError:
        return 0.0  # fallback if no vocab
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(X[0], X[1])[0, 0]

def find_weak_spans(text: str) -> List[Dict]:
    """Simple heuristic: long sentences or very generic phrases."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    spans = []
    sentences = sent_tokenize(text or "")
    cursor = 0
    for s in sentences:
        start = text.find(s, cursor)
        end = start + len(s)
        cursor = end
        if len(s.split()) > 28 or tfidf_similarity(s, "introduction overview general common things") > 0.35:
            spans.append({"span": [start, end], "issue": "low specificity / long sentence", "fix_hint": "shorten and add concrete detail"})
    return spans
