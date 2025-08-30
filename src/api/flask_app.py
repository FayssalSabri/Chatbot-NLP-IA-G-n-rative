from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import settings
from data.preprocessing import ReviewPreprocessor
from models.vector_search import SemanticSearch
from business_rules.rules_engine import BusinessRulesEngine
from models.gpt_finetuning import GPTFineTuner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
preprocessor = ReviewPreprocessor()
semantic_search = SemanticSearch()
rules_engine = BusinessRulesEngine()

# Try to load existing search index
try:
    semantic_search.load(settings.MODELS_DIR)
    logger.info("Loaded existing search index")
except:
    logger.info("No existing search index found")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'components': {
            'preprocessor': 'ready',
            'semantic_search': 'ready',
            'rules_engine': 'ready'
        }
    })

@app.route('/preprocess', methods=['POST'])
def preprocess_text():
    """Preprocess text endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Clean text
        cleaned_text = preprocessor.clean_text(text)
        
        # Extract features
        sentiment = preprocessor.extract_sentiment(cleaned_text)
        entities = preprocessor.extract_entities(cleaned_text)
        aspects = preprocessor.extract_aspects(cleaned_text)
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'entities': entities,
            'aspects': aspects
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def semantic_search_endpoint():
    """Semantic search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 10)
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Perform search
        results = semantic_search.search(query, k=k, filters=filters)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_review():
    """Complete analysis endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        rating = data.get('rating', None)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Preprocess
        cleaned_text = preprocessor.clean_text(text)
        sentiment = preprocessor.extract_sentiment(cleaned_text)
        entities = preprocessor.extract_entities(cleaned_text)
        aspects = preprocessor.extract_aspects(cleaned_text)
        
        # Create context for rules engine
        context = {
            'text': cleaned_text,
            'rating': rating,
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'aspects': aspects,
            'entities': entities
        }
        
        # Apply business rules
        rule_results = rules_engine.process(context)
        
        # Execute actions
        execution_summary = rules_engine.execute_actions(rule_results)
        
        # Search for similar reviews
        similar_reviews = semantic_search.search(cleaned_text, k=5)
        
        result = {
            'analysis': {
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'entities': entities,
                'aspects': aspects
            },
            'business_rules': {
                'triggered_rules': [r.rule_name for r in rule_results],
                'execution_summary': execution_summary
            },
            'similar_reviews': similar_reviews
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rules', methods=['GET'])
def get_rules():
    """Get business rules"""
    try:
        stats = rules_engine.get_rule_statistics()
        rules_info = []
        
        for rule in rules_engine.rules:
            rules_info.append({
                'name': rule.name,
                'description': rule.description,
                'priority': rule.priority.name,
                'active': rule.active,
                'conditions_count': len(rule.conditions),
                'actions_count': len(rule.actions)
            })
        
        return jsonify({
            'statistics': stats,
            'rules': rules_info
        })
        
    except Exception as e:
        logger.error(f"Rules error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/index_documents', methods=['POST'])
def index_documents():
    """Index documents for search"""
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        
        if not documents:
            return jsonify({'error': 'Documents list is required'}), 400
        
        # Prepare metadata
        metadata = []
        texts = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                texts.append(doc)
                metadata.append({'id': i, 'type': 'review'})
            else:
                texts.append(doc.get('text', ''))
                meta = {k: v for k, v in doc.items() if k != 'text'}
                meta['id'] = i
                metadata.append(meta)
        
        # Index documents
        semantic_search.index_documents(texts, metadata)
        
        # Save index
        semantic_search.save(settings.MODELS_DIR)
        
        return jsonify({
            'message': f'Indexed {len(texts)} documents',
            'count': len(texts)
        })
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG
    )