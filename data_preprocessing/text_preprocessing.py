import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download stopwords if you haven't already
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """
    Clean and preprocess the text data.
    This includes removing special characters, stopwords, and tokenizing.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        processed_text (str): The cleaned and tokenized text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(f"[{string.punctuation}]", "", text)
    
    # Tokenize the text (split it into words)
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Return the cleaned text (as a list of words or a single string)
    return " ".join(tokens)

