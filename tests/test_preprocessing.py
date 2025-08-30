import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocessing import ReviewPreprocessor

class TestReviewPreprocessor:
    
    @pytest.fixture
    def preprocessor(self):
        return ReviewPreprocessor()
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "Le service client était excellent et très réactif !",
            "Produit de mauvaise qualité, très déçu de mon achat.",
            "Prix un peu élevé mais la qualité est au rendez-vous.",
            "Livraison rapide et produit conforme à la description."
        ]
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning functionality"""
        # Test basic cleaning
        text = "Bonjour ! Voici un texte avec des MAJUSCULES et des émojis 😊"
        cleaned = preprocessor.clean_text(text)
        
        assert cleaned.islower()
        assert "😊" not in cleaned
        assert cleaned.strip() != ""
    
    def test_clean_text_urls(self, preprocessor):
        """Test URL removal"""
        text = "Visitez https://example.com pour plus d'infos"
        cleaned = preprocessor.clean_text(text)
        
        assert "https://example.com" not in cleaned
        assert "visitez" in cleaned
    
    def test_clean_text_emails(self, preprocessor):
        """Test email removal"""
        text = "Contactez-nous à contact@example.com"
        cleaned = preprocessor.clean_text(text)
        
        assert "contact@example.com" not in cleaned
        assert "contactez" in cleaned
    
    def test_extract_sentiment(self, preprocessor, sample_texts):
        """Test sentiment extraction"""
        for text in sample_texts:
            sentiment = preprocessor.extract_sentiment(text)
            
            assert 'polarity' in sentiment
            assert 'subjectivity' in sentiment
            assert -1 <= sentiment['polarity'] <= 1
            assert 0 <= sentiment['subjectivity'] <= 1
    
    def test_extract_aspects(self, preprocessor):
        """Test aspect extraction"""
        # Test service aspect
        text = "Le service client était décevant"
        aspects = preprocessor.extract_aspects(text)
        assert 'service' in aspects
        
        # Test product aspect
        text = "Le produit est de mauvaise qualité"
        aspects = preprocessor.extract_aspects(text)
        assert any(aspect in aspects for aspect in ['produit', 'qualité'])
        
        # Test delivery aspect
        text = "La livraison était très rapide"
        aspects = preprocessor.extract_aspects(text)
        assert 'livraison' in aspects
    
    def test_extract_entities(self, preprocessor):
        """Test entity extraction"""
        if preprocessor.nlp is None:
            pytest.skip("spaCy model not available")
        
        text = "J'ai commandé chez Amazon le 15 janvier"
        entities = preprocessor.extract_entities(text)
        
        assert isinstance(entities, list)
        # Check that each entity has required fields
        for entity in entities:
            assert 'text' in entity
            assert 'label' in entity
            assert 'start' in entity
            assert 'end' in entity
    
    def test_preprocess_reviews_dataframe(self, preprocessor, sample_texts):
        """Test complete preprocessing pipeline"""
        df = pd.DataFrame({'text': sample_texts})
        
        processed_df = preprocessor.preprocess_reviews(df)
        
        # Check all required columns are present
        required_columns = [
            'cleaned_text', 'sentiment_polarity', 'sentiment_subjectivity',
            'entities', 'aspects'
        ]
        
        for col in required_columns:
            assert col in processed_df.columns
        
        # Check data integrity
        assert len(processed_df) == len(sample_texts)
        assert not processed_df['cleaned_text'].isnull().any()
        assert not processed_df['sentiment_polarity'].isnull().any()
    
    def test_empty_text_handling(self, preprocessor):
        """Test handling of empty or invalid text"""
        # Test empty string
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
        
        # Test sentiment extraction with empty text
        sentiment = preprocessor.extract_sentiment("")
        assert sentiment['polarity'] == 0
        assert sentiment['subjectivity'] == 0
        
        # Test aspects extraction with empty text
        aspects = preprocessor.extract_aspects("")
        assert aspects == []
