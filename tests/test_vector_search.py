import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.vector_search import FAISSVectorSearch, SemanticSearch

class TestFAISSVectorSearch:
    
    @pytest.fixture
    def vector_search(self):
        return FAISSVectorSearch(embedding_dim=384, index_type="flat")
    
    @pytest.fixture
    def sample_embeddings(self):
        np.random.seed(42)
        return np.random.random((10, 384)).astype('float32')
    
    @pytest.fixture
    def sample_metadata(self):
        return [
            {'id': i, 'text': f'Document {i}', 'category': 'test'}
            for i in range(10)
        ]
    
    def test_create_index(self, vector_search):
        """Test index creation"""
        assert vector_search.index is not None
        assert vector_search.embedding_dim == 384
        assert vector_search.index_type == "flat"
    
    def test_add_documents(self, vector_search, sample_embeddings, sample_metadata):
        """Test adding documents to index"""
        vector_search.add_documents(sample_embeddings, sample_metadata)
        
        assert vector_search.index.ntotal == len(sample_embeddings)
        assert len(vector_search.metadata) == len(sample_metadata)
    
    def test_search(self, vector_search, sample_embeddings, sample_metadata):
        """Test search functionality"""
        # Add documents
        vector_search.add_documents(sample_embeddings, sample_metadata)
        
        # Search with first embedding
        query = sample_embeddings[0:1]
        scores, results = vector_search.search(query, k=3)
        
        assert len(scores) <= 3
        assert len(results) <= 3
        assert len(scores) == len(results)
        
        # First result should be the exact match (distance ≈ 0)
        assert scores[0] < 0.01  # Very small distance
        assert results[0]['id'] == 0
    
    def test_search_with_filters(self, vector_search, sample_embeddings):
        """Test search with metadata filters"""
        # Add documents with different categories
        metadata = [
            {'id': i, 'category': 'A' if i < 5 else 'B', 'text': f'Doc {i}'}
            for i in range(10)
        ]
        
        vector_search.add_documents(sample_embeddings, metadata)
        
        # Search with category filter
        query = sample_embeddings[0:1]
        filters = {'category': 'A'}
        scores, results = vector_search.search_with_filters(query, filters, k=3)
        
        # All results should be from category A
        for result in results:
            assert result['category'] == 'A'
    
    def test_save_and_load_index(self, vector_search, sample_embeddings, sample_metadata):
        """Test saving and loading index"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            index_path = tmp_path / "test.index"
            metadata_path = tmp_path / "test_metadata.pkl"
            
            # Add documents and save
            vector_search.add_documents(sample_embeddings, sample_metadata)
            vector_search.save_index(index_path, metadata_path)
            
            # Create new instance and load
            new_search = FAISSVectorSearch(embedding_dim=384)
            new_search.load_index(index_path, metadata_path)
            
            assert new_search.index.ntotal == vector_search.index.ntotal
            assert len(new_search.metadata) == len(vector_search.metadata)
    
    def test_get_stats(self, vector_search, sample_embeddings, sample_metadata):
        """Test statistics retrieval"""
        stats = vector_search.get_stats()
        assert stats['total_documents'] == 0
        
        vector_search.add_documents(sample_embeddings, sample_metadata)
        stats = vector_search.get_stats()
        
        assert stats['total_documents'] == len(sample_embeddings)
        assert stats['embedding_dimension'] == 384
        assert stats['index_type'] == "flat"

class TestSemanticSearch:
    
    @pytest.fixture
    def semantic_search(self):
        # Use a smaller model for testing
        return SemanticSearch("sentence-transformers/all-MiniLM-L6-v2")
    
    @pytest.fixture
    def sample_documents(self):
        return [
            "Le service client était excellent",
            "Produit de mauvaise qualité",
            "Livraison rapide et efficace",
            "Prix très élevé pour la qualité",
            "Interface utilisateur intuitive"
        ]
    
    @pytest.fixture
    def sample_metadata(self):
        return [
            {'id': i, 'category': 'review', 'rating': rating}
            for i, rating in enumerate([5, 2, 4, 2, 4])
        ]
    
    def test_index_documents(self, semantic_search, sample_documents, sample_metadata):
        """Test document indexing"""
        semantic_search.index_documents(sample_documents, sample_metadata)
        
        stats = semantic_search.vector_search.get_stats()
        assert stats['total_documents'] == len(sample_documents)
    
    def test_search_semantic(self, semantic_search, sample_documents, sample_metadata):
        """Test semantic search"""
        # Index documents
        semantic_search.index_documents(sample_documents, sample_metadata)
        
        # Search for similar content
        results = semantic_search.search("problème avec le service", k=3)
        
        assert len(results) <= 3
        assert all('similarity_score' in result for result in results)
        assert all('relevance_score' in result for result in results)
        
        # Results should be ordered by relevance
        scores = [result['relevance_score'] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_filters(self, semantic_search, sample_documents, sample_metadata):
        """Test search with metadata filters"""
        semantic_search.index_documents(sample_documents, sample_metadata)
        
        # Search with rating filter
        filters = {'rating': 4}
        results = semantic_search.search("produit", k=5, filters=filters)
        
        # All results should have rating 4
        for result in results:
            assert result.get('rating') == 4
    
    def test_save_and_load(self, semantic_search, sample_documents, sample_metadata):
        """Test saving and loading semantic search"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Index documents and save
            semantic_search.index_documents(sample_documents, sample_metadata)
            semantic_search.save(tmp_path)
            
            # Create new instance and load
            new_search = SemanticSearch()
            new_search.load(tmp_path)
            
            # Test that loaded search works
            results = new_search.search("service", k=2)
            assert len(results) <= 2
