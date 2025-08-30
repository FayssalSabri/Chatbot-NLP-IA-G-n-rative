import pandas as pd
import re
import spacy
from typing import List, Dict, Any
from textblob import TextBlob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.error("French spaCy model not found. Run: python -m spacy download fr_core_news_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep accents
        text = re.sub(r'[^\w\s\À-ÿ]', ' ', text)
        
        return text.lower()
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_aspects(self, text: str) -> List[str]:
        """Extract product/service aspects"""
        # Define aspect keywords for customer reviews
        aspect_keywords = {
            'service': ['service', 'accueil', 'personnel', 'équipe', 'conseiller'],
            'qualité': ['qualité', 'qualite', 'bien', 'bon', 'mauvais', 'excellent'],
            'prix': ['prix', 'coût', 'cout', 'cher', 'économique', 'gratuit'],
            'livraison': ['livraison', 'délai', 'delai', 'rapide', 'lent', 'expédition'],
            'produit': ['produit', 'article', 'item', 'commande', 'achat']
        }
        
        detected_aspects = []
        text_lower = text.lower()
        
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_aspects.append(aspect)
        
        return detected_aspects
    
    def preprocess_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        logger.info(f"Preprocessing {len(reviews_df)} reviews...")
        
        # Clean text
        reviews_df['cleaned_text'] = reviews_df['text'].apply(self.clean_text)
        
        # Extract sentiment
        sentiment_data = reviews_df['cleaned_text'].apply(self.extract_sentiment)
        reviews_df['sentiment_polarity'] = sentiment_data.apply(lambda x: x['polarity'])
        reviews_df['sentiment_subjectivity'] = sentiment_data.apply(lambda x: x['subjectivity'])
        
        # Extract entities
        reviews_df['entities'] = reviews_df['cleaned_text'].apply(self.extract_entities)
        
        # Extract aspects
        reviews_df['aspects'] = reviews_df['cleaned_text'].apply(self.extract_aspects)
        
        logger.info("Preprocessing completed!")
        return reviews_df

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = {
        'text': [
            "Le service client était excellent et la livraison très rapide!",
            "Produit de mauvaise qualité, très déçu de mon achat.",
            "Prix un peu élevé mais la qualité est au rendez-vous."
        ],
        'rating': [5, 2, 4]
    }
    
    df = pd.DataFrame(sample_data)
    preprocessor = ReviewPreprocessor()
    processed_df = preprocessor.preprocess_reviews(df)
    
    print(processed_df[['text', 'sentiment_polarity', 'aspects']].head())
