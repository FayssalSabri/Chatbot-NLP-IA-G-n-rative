import chainlit as cl
import asyncio
from pathlib import Path
import sys
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import settings
from data.preprocessing import ReviewPreprocessor
from models.vector_search import SemanticSearch
from business_rules.rules_engine import BusinessRulesEngine
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
preprocessor = ReviewPreprocessor()
semantic_search = SemanticSearch()
rules_engine = BusinessRulesEngine()
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Try to load existing search index
try:
    semantic_search.load(settings.MODELS_DIR)
    logger.info("Loaded existing search index")
except:
    logger.info("No existing search index found")

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    await cl.Message(
        content="""# 🤖 Assistant IA Analyse d'Avis Clients

Bonjour ! Je suis votre assistant intelligent spécialisé dans l'analyse des avis clients. 

## 🎯 Mes capacités :
- **Analyse sentiment** : Je comprends les émotions dans les avis
- **Recherche sémantique** : Je trouve des avis similaires rapidement
- **Règles métier** : J'applique des règles automatiques selon le contexte
- **Recommandations** : Je propose des actions adaptées

## 📝 Comment m'utiliser :
1. **Tapez un avis client** pour une analyse complète
2. **Posez une question** sur les avis existants
3. **Demandez des statistiques** sur les données
4. **Explorez les fonctionnalités** avec des commandes

*Essayez par exemple : "Analyse cet avis : Le service était décevant mais le produit correct"*
        """,
        author="Assistant"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_input = message.content.strip()
    
    # Show thinking indicator
    msg = cl.Message(content="", author="Assistant")
    await msg.send()
    
    try:
        # Determine the type of request
        if user_input.lower().startswith(("analyse", "analyser")):
            response = await handle_analysis_request(user_input, msg)
        elif user_input.lower().startswith(("recherche", "cherche", "trouve")):
            response = await handle_search_request(user_input, msg)
        elif user_input.lower().startswith(("statistiques", "stats", "règles")):
            response = await handle_stats_request(user_input, msg)
        elif "aide" in user_input.lower() or "help" in user_input.lower():
            response = await handle_help_request(msg)
        else:
            # Default: treat as review analysis
            response = await handle_review_analysis(user_input, msg)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await msg.update(content=f"❌ **Erreur**: {str(e)}")

async def handle_analysis_request(user_input: str, msg: cl.Message):
    """Handle explicit analysis requests"""
    # Extract text after "analyse" command
    text_to_analyze = user_input.split(":", 1)[-1].strip() if ":" in user_input else user_input
    
    await msg.update(content="🔍 **Analyse en cours...**")
    
    # Perform analysis
    analysis_result = await perform_complete_analysis(text_to_analyze)
    
    # Format response
    response = format_analysis_response(analysis_result)
    await msg.update(content=response)

async def handle_search_request(user_input: str, msg: cl.Message):
    """Handle search requests"""
    # Extract search query
    query = user_input.split(":", 1)[-1].strip() if ":" in user_input else user_input
    
    await msg.update(content="🔍 **Recherche en cours...**")
    
    try:
        # Perform semantic search
        results = semantic_search.search(query, k=5)
        
        if not results:
            response = f"❌ **Aucun résultat trouvé** pour : *{query}*"
        else:
            response = f"🔍 **Résultats de recherche** pour : *{query}*\n\n"
            for i, result in enumerate(results, 1):
                score = result.get('relevance_score', 0)
                response += f"**{i}.** Score: {score:.3f}\n"
                response += f"_{result.get('text', 'Texte non disponible')[:100]}..._\n\n"
        
        await msg.update(content=response)
        
    except Exception as e:
        await msg.update(content=f"❌ **Erreur de recherche**: {str(e)}")

async def handle_stats_request(user_input: str, msg: cl.Message):
    """Handle statistics requests"""
    await msg.update(content="📊 **Génération des statistiques...**")
    
    try:
        # Get rules statistics
        rules_stats = rules_engine.get_rule_statistics()
        search_stats = semantic_search.vector_search.get_stats()
        
        response = f"""📊 **Statistiques du Système**

## 🎯 Règles Métier
- **Total des règles**: {rules_stats['total_rules']}
- **Règles actives**: {rules_stats['active_rules']}

### Répartition par priorité :
"""
        
        for priority, count in rules_stats['rules_by_priority'].items():
            response += f"- **{priority}**: {count} règles\n"
        
        response += f"""

## 🔍 Base de Recherche
- **Documents indexés**: {search_stats['total_documents']}
- **Dimension des embeddings**: {search_stats['embedding_dimension']}
- **Type d'index**: {search_stats['index_type']}
"""
        
        await msg.update(content=response)
        
    except Exception as e:
        await msg.update(content=f"❌ **Erreur statistiques**: {str(e)}")

async def handle_help_request(msg: cl.Message):
    """Handle help requests"""
    help_content = """# 📚 Guide d'utilisation

## 🎯 Commandes principales :

### 📝 Analyse d'avis
- **`Analyse: [votre texte]`** - Analyse complète d'un avis
- Ou tapez directement un avis pour l'analyser

### 🔍 Recherche
- **`Recherche: [votre requête]`** - Trouve des avis similaires
- **`Cherche: [mots-clés]`** - Recherche sémantique

### 📊 Statistiques
- **`Statistiques`** - Affiche les stats du système
- **`Règles`** - Information sur les règles métier

## 💡 Exemples d'utilisation :

```
Analyse: Le service était décevant mais la livraison rapide

Recherche: problèmes de qualité produit

Statistiques
```

## 🎯 Fonctionnalités :
- ✅ Analyse de sentiment automatique
- ✅ Détection d'aspects produit/service
- ✅ Application de règles métier
- ✅ Recherche sémantique intelligente
- ✅ Recommandations d'actions
"""
    
    await msg.update(content=help_content)

async def handle_review_analysis(text: str, msg: cl.Message):
    """Handle direct review analysis"""
    await msg.update(content="🔍 **Analyse de l'avis en cours...**")
    
    # Perform complete analysis
    analysis_result = await perform_complete_analysis(text)
    
    # Format and send response
    response = format_analysis_response(analysis_result)
    await msg.update(content=response)

async def perform_complete_analysis(text: str) -> Dict[str, Any]:
    """Perform complete analysis of text"""
    # Preprocess
    cleaned_text = preprocessor.clean_text(text)
    sentiment = preprocessor.extract_sentiment(cleaned_text)
    entities = preprocessor.extract_entities(cleaned_text)
    aspects = preprocessor.extract_aspects(cleaned_text)
    
    # Create context for rules engine
    context = {
        'text': cleaned_text,
        'sentiment_polarity': sentiment['polarity'],
        'sentiment_subjectivity': sentiment['subjectivity'],
        'aspects': aspects,
        'entities': entities
    }
    
    # Apply business rules
    rule_results = rules_engine.process(context)
    execution_summary = rules_engine.execute_actions(rule_results)
    
    # Search for similar reviews
    try:
        similar_reviews = semantic_search.search(cleaned_text, k=3)
    except:
        similar_reviews = []
    
    # Generate AI response using GPT
    ai_response = await generate_ai_response(context, rule_results)
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'entities': entities,
        'aspects': aspects,
        'rule_results': rule_results,
        'execution_summary': execution_summary,
        'similar_reviews': similar_reviews,
        'ai_response': ai_response
    }

async def generate_ai_response(context: Dict[str, Any], rule_results) -> str:
    """Generate AI response using GPT"""
    try:
        # Create prompt
        prompt = f"""
Analysez cet avis client et fournissez une réponse empathique et professionnelle.

Avis: {context['text']}
Sentiment: {context['sentiment_polarity']:.2f} (polarité)
Aspects détectés: {', '.join(context['aspects'])}
Règles déclenchées: {[r.rule_name for r in rule_results]}

Fournissez une réponse qui :
1. Montre de l'empathie envers le client
2. Adresse les points spécifiques mentionnés
3. Propose des solutions concrètes si nécessaire
4. Reste professionnelle et constructive
"""
        
        response = openai_client.chat.completions.create(
            model=settings.GPT_MODEL,
            messages=[
                {"role": "system", "content": "Vous êtes un assistant de service client expert, empathique et professionnel."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "Merci pour votre retour. Nous prenons vos commentaires au sérieux et nous nous efforçons d'améliorer continuellement nos services."

def format_analysis_response(analysis: Dict[str, Any]) -> str:
    """Format analysis results into readable response"""
    sentiment = analysis['sentiment']
    aspects = analysis['aspects']
    rule_results = analysis['rule_results']
    similar_count = len(analysis['similar_reviews'])
    
    # Determine sentiment emoji and description
    polarity = sentiment['polarity']
    if polarity > 0.1:
        sentiment_emoji = "😊"
        sentiment_desc = "Positif"
    elif polarity < -0.1:
        sentiment_emoji = "😞"
        sentiment_desc = "Négatif"
    else:
        sentiment_emoji = "😐"
        sentiment_desc = "Neutre"
    
    response = f"""# 📋 Analyse Complète

##  **Avis Original**
*{analysis['original_text']}*

##  **Analyse Sentiment** {sentiment_emoji}
- **Polarité**: {polarity:.2f} ({sentiment_desc})
- **Subjectivité**: {sentiment['subjectivity']:.2f}

##  **Aspects Détectés**
{', '.join(f'`{aspect}`' for aspect in aspects) if aspects else '*Aucun aspect spécifique détecté*'}

##  **Règles Déclenchées**
{chr(10).join(f'- **{r.rule_name}**: {len(r.actions_triggered)} action(s)' for r in rule_results) if rule_results else '*Aucune règle déclenchée*'}

##  **Réponse IA Recommandée**
{analysis['ai_response']}

##  **Avis Similaires Trouvés**
*{similar_count} avis similaires dans la base de données*

---
 *Analyse générée par l'IA - Système de traitement des avis clients*
"""
    
    return response

# Chainlit settings
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="analyst",
            markdown_description="Assistant d'analyse des avis clients",
            icon="https://picsum.photos/200",
        ),
    ]
