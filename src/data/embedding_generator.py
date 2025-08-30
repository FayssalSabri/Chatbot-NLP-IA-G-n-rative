import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model"""
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and isinstance(text, str)]
        
        if not valid_texts:
            logger.warning("No valid texts found for embedding generation")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                valid_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], 
                       file_path: Path) -> None:
        """Save embeddings and metadata to disk"""
        data = {
            'embeddings': embeddings,
            'texts': texts,
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Embeddings saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self, file_path: Path) -> tuple:
        """Load embeddings and metadata from disk"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded embeddings from {file_path}")
            return data['embeddings'], data['texts'], data.get('model_name', 'unknown')
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """Process a DataFrame and add embeddings column"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].fillna('').tolist()
        embeddings = self.generate_embeddings(texts)
        
        if len(embeddings) > 0:
            df['embeddings'] = list(embeddings)
        else:
            df['embeddings'] = [np.array([])] * len(df)
        
        return df

# Example usage
if __name__ == "__main__":
    from src.utils.config import settings
    
    # Sample data
    sample_texts = [
        "Le service client était excellent et la livraison très rapide!",
        "Produit de mauvaise qualité, très déçu de mon achat.",
        "Prix un peu élevé mais la qualité est au rendez-vous."
    ]
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator()
    embeddings = embedding_gen.generate_embeddings(sample_texts)
    
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Save embeddings
    save_path = settings.MODELS_DIR / "sample_embeddings.pkl"
    embedding_gen.save_embeddings(embeddings, sample_texts, save_path)
