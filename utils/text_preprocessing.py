"""
utils/text_preprocessing.py
----------------------------
Handles all NLP preprocessing steps:
  - Text cleaning
  - Tokenization
  - Stopword removal
  - Lemmatization
"""

import re
import string

# We use only stdlib + lightweight deps so this runs without heavy installs.
# If NLTK is available we use it; otherwise we fall back to a simple splitter.

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data (silent if already present)
    NLTK_DATA_DOWNLOADED = False
    for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
            NLTK_DATA_DOWNLOADED = True
        except Exception as e:
            print(f"Warning: Failed to download NLTK data '{pkg}': {e}")
            pass

    NLTK_AVAILABLE = NLTK_DATA_DOWNLOADED
    print(f"NLTK available: {NLTK_AVAILABLE}")
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available, using fallback text processing")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Lowercase
      - Remove URLs, emails, special chars
      - Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                   # remove emails
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)          # keep alphanum + # +
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split text into word tokens."""
    if NLTK_AVAILABLE:
        return word_tokenize(text)
    # Fallback: simple whitespace split
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords."""
    if NLTK_AVAILABLE:
        stop_words = set(stopwords.words("english"))
    else:
        # Minimal fallback stopword list
        stop_words = {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
            "it", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "shall", "can", "a", "an", "the", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "up", "down", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "who", "which",
            "this", "that", "these", "those", "not", "no", "nor", "so", "yet",
            "both", "either", "neither", "such", "more", "also", "just", "about",
        }
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    """Reduce words to their base form."""
    if NLTK_AVAILABLE:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    # Fallback: strip trailing 's' as naive lemmatization
    return [t.rstrip("s") if len(t) > 3 else t for t in tokens]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(text: str, return_tokens: bool = False):
    """
    Full NLP preprocessing pipeline.

    Args:
        text: Raw input text (resume or job description).
        return_tokens: If True, return list of tokens; else return joined string.

    Returns:
        Processed tokens (list) or cleaned string.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    if return_tokens:
        return tokens
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    Experienced Data Scientist with 4 years of experience in Python, Machine Learning,
    TensorFlow and SQL. Built recommendation systems and NLP pipelines.
    Contact: john@example.com | https://linkedin.com/in/john
    """
    print("Raw text:", sample[:80], "...")
    print("\nCleaned:", clean_text(sample)[:80], "...")
    tokens = preprocess(sample, return_tokens=True)
    print("\nFinal tokens:", tokens)
