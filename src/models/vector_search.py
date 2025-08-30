import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FAISSVectorSearch:
    def __init__(self, embedding_dim: int = 384, index_type: str = "flat"):
        """Initialize FAISS vector search"""
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.embedding_model = None
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type}")
    
    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add documents to the index"""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} documents to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[float], List[Dict]]:
        """Search for similar documents"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get results
        results = []
        scores = []
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
                scores.append(float(distance))
        
        return scores, results
    
    def search_with_filters(self, query_embedding: np.ndarray, 
                          filters: Dict = None, k: int = 10) -> Tuple[List[float], List[Dict]]:
        """Search with metadata filters"""
        scores, results = self.search(query_embedding, k * 3)  # Get more results for filtering
        
        if not filters:
            return scores[:k], results[:k]
        
        # Apply filters
        filtered_scores = []
        filtered_results = []
        
        for score, result in zip(scores, results):
            if self._matches_filters(result, filters):
                filtered_scores.append(score)
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
        
        return filtered_scores, filtered_results
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def save_index(self, index_path: Path, metadata_path: Path):
        """Save index and metadata"""
        try:
            faiss.write_index(self.index, str(index_path))
            
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type
                }, f)
            
            logger.info(f"Index saved to {index_path}, metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, index_path: Path, metadata_path: Path):
        """Load index and metadata"""
        try:
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.embedding_dim = data['embedding_dim']
                self.index_type = data['index_type']
            
            logger.info(f"Index loaded from {index_path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_documents': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata)
        }

class SemanticSearch:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Semantic search combining embeddings and FAISS"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_search = FAISSVectorSearch(
            embedding_dim=self.embedding_model.get_sentence_embedding_dimension()
        )
        
    def index_documents(self, documents: List[str], metadata: List[Dict]):
        """Index documents for semantic search"""
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Add to FAISS index
        self.vector_search.add_documents(embeddings, metadata)
        
    def search(self, query: str, k: int = 10, filters: Dict = None) -> List[Dict]:
        """Semantic search for query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        if filters:
            scores, results = self.vector_search.search_with_filters(query_embedding, filters, k)
        else:
            scores, results = self.vector_search.search(query_embedding, k)
        
        # Combine scores with results
        for i, result in enumerate(results):
            result['similarity_score'] = scores[i]
            result['relevance_score'] = 1 / (1 + scores[i])  # Convert distance to relevance
        
        return results
    
    def save(self, base_path: Path):
        """Save the search index"""
        index_path = base_path / "faiss.index"
        metadata_path = base_path / "metadata.pkl"
        self.vector_search.save_index(index_path, metadata_path)
    
    def load(self, base_path: Path):
        """Load the search index"""
        index_path = base_path / "faiss.index"
        metadata_path = base_path / "metadata.pkl"
        self.vector_search.load_index(index_path, metadata_path)

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Le service client était excellent et très réactif",
        "Produit de mauvaise qualité, très déçu",
        "Livraison rapide et produit conforme à la description",
        "Prix élevé mais qualité au rendez-vous",
        "Interface utilisateur intuitive et facile à utiliser"
    ]
    
    metadata = [
        {"id": i, "rating": rating, "category": "review"} 
        for i, rating in enumerate([5, 2, 4, 4, 5])
    ]
    
    # Create semantic search
    search_engine = SemanticSearch()
    
    # Index documents
    search_engine.index_documents(documents, metadata)
    
    # Search
    results = search_engine.search("problème avec la qualité du produit", k=3)
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: Score={result['relevance_score']:.3f}, Rating={result['rating']}")
