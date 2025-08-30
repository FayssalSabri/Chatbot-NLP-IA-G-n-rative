"""Generate sample data for testing the chatbot system"""

import pandas as pd
import random
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import settings

def generate_sample_reviews(num_reviews: int = 200) -> pd.DataFrame:
    """Generate sample customer reviews"""
    
    # Sample review templates
    positive_templates = [
        "Excellent service, très satisfait de mon achat ! La livraison était rapide et le produit de qualité.",
        "Super produit, conforme à mes attentes. L'équipe client est très professionnelle.",
        "Très bonne expérience, je recommande vivement ! Prix correct et service impeccable.",
        "Parfait ! Livraison dans les temps et produit exactement comme décrit.",
        "Service client au top, ils ont résolu mon problème rapidement. Très content !",
    ]
    
    negative_templates = [
        "Très déçu de mon achat. Le produit ne correspond pas à la description et la qualité est mauvaise.",
        "Service client décevant, personne ne répond au téléphone. Livraison en retard.",
        "Mauvaise qualité pour le prix payé. Je ne recommande pas ce produit.",
        "Problème avec la commande, produit défectueux reçu. Difficile de se faire rembourser.",
        "Service après-vente inexistant. Très mauvaise expérience, je ne recommande pas.",
    ]
    
    neutral_templates = [
        "Produit correct sans plus. Rien d'exceptionnel mais fait le travail.",
        "Service standard, pas de problème particulier mais rien d'extraordinaire non plus.",
        "Livraison dans les temps, produit conforme. Expérience normale.",
        "Prix un peu élevé mais qualité acceptable. Service client moyen.",
        "Produit correct, correspond à ce qui était annoncé. Délai de livraison respecté.",
    ]
    
    # Generate reviews
    reviews = []
    
    for i in range(num_reviews):
        # Randomly choose sentiment
        sentiment_type = random.choices(
            ['positive', 'negative', 'neutral'],
            weights=[0.4, 0.3, 0.3]  # 40% positive, 30% negative, 30% neutral
        )[0]
        
        if sentiment_type == 'positive':
            text = random.choice(positive_templates)
            rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
        elif sentiment_type == 'negative':
            text = random.choice(negative_templates)
            rating = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        else:  # neutral
            text = random.choice(neutral_templates)
            rating = 3
        
        # Add some variation to the text
        variations = [
            " La commande est arrivée rapidement.",
            " Le packaging était soigné.",
            " J'ai eu quelques questions et le support a bien répondu.",
            " Bon rapport qualité-prix.",
            " Interface du site facile à utiliser.",
            " Délai de livraison respecté.",
        ]
        
        if random.random() < 0.3:  # 30% chance to add variation
            text += random.choice(variations)
        
        # Generate metadata
        review = {
            'id': i + 1,
            'text': text,
            'rating': rating,
            'sentiment_type': sentiment_type,
            'customer_id': f'CUST_{random.randint(1000, 9999)}',
            'product_category': random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books']),
            'date': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
            'verified_purchase': random.choice([True, False], k=1)[0]
        }
        
        reviews.append(review)
    
    return pd.DataFrame(reviews)

def generate_training_responses(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Generate training responses for fine-tuning"""
    
    response_templates = {
        'positive': [
            "Merci beaucoup pour ce retour positif ! Nous sommes ravis que vous soyez satisfait de votre achat et de notre service. N'hésitez pas à nous contacter si vous avez besoin d'aide.",
            "C'est formidable d'entendre que vous êtes content ! Votre satisfaction est notre priorité. Merci de nous faire confiance.",
            "Merci pour ces commentaires encourageants ! Nous continuerons à faire de notre mieux pour maintenir cette qualité de service."
        ],
        'negative': [
            "Nous sommes sincèrement désolés pour cette expérience décevante. Vos commentaires sont précieux et nous allons examiner ces points pour nous améliorer. Pouvons-nous vous contacter pour résoudre ce problème ?",
            "Je comprends votre frustration et nous prenons vos préoccupations très au sérieux. Notre équipe va étudier votre cas en priorité. Merci de nous donner l'opportunité de corriger cette situation.",
            "Nous nous excusons pour les désagréments causés. Cette expérience ne reflète pas nos standards habituels. Nous aimerions rectifier la situation rapidement."
        ],
        'neutral': [
            "Merci pour votre retour. Nous apprécions que vous ayez pris le temps de partager votre expérience. Si vous avez des suggestions d'amélioration, nous serions ravis de les entendre.",
            "Nous prenons note de vos commentaires. Notre objectif est de dépasser vos attentes et nous travaillons constamment à améliorer nos services.",
            "Merci pour ce retour constructif. Nous restons à votre disposition si vous avez des questions ou des besoins spécifiques."
        ]
    }
    
    training_data = []
    
    for _, review in reviews_df.iterrows():
        sentiment_type = review['sentiment_type']
        text = review['text']
        rating = review['rating']
        
        # Choose appropriate response
        response = random.choice(response_templates[sentiment_type])
        
        # Add specific elements based on review content
        if 'livraison' in text.lower() and sentiment_type == 'negative':
            response += " Notre équipe logistique sera informée pour éviter ce type de problème à l'avenir."
        elif 'qualité' in text.lower() and sentiment_type == 'negative':
            response += " Nous allons transmettre vos remarques à notre équipe qualité."
        elif 'service' in text.lower() and sentiment_type == 'negative':
            response += " Nous allons former notre équipe pour améliorer l'expérience client."
        
        training_data.append({
            'review_id': review['id'],
            'text': text,
            'rating': rating,
            'response': response,
            'sentiment_type': sentiment_type
        })
    
    return pd.DataFrame(training_data)

def main():
    """Generate and save sample data"""
    print("🚀 Génération des données d'exemple...")
    
    # Create directories
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate sample reviews
    print("📝 Génération des avis clients...")
    reviews_df = generate_sample_reviews(200)
    
    # Generate training responses
    print("🤖 Génération des réponses d'entraînement...")
    training_df = generate_training_responses(reviews_df)
    
    # Save data
    reviews_path = settings.RAW_DATA_DIR / "sample_reviews.csv"
    training_path = settings.RAW_DATA_DIR / "training_responses.csv"
    
    reviews_df.to_csv(reviews_path, index=False, encoding='utf-8')
    training_df.to_csv(training_path, index=False, encoding='utf-8')
    
    print(f"✅ Données sauvegardées :")
    print(f"   - Avis clients : {reviews_path} ({len(reviews_df)} avis)")
    print(f"   - Réponses d'entraînement : {training_path} ({len(training_df)} exemples)")
    
    # Generate summary statistics
    print(f"\n📊 Statistiques :")
    print(f"   - Avis positifs : {len(reviews_df[reviews_df['sentiment_type'] == 'positive'])}")
    print(f"   - Avis négatifs : {len(reviews_df[reviews_df['sentiment_type'] == 'negative'])}")
    print(f"   - Avis neutres : {len(reviews_df[reviews_df['sentiment_type'] == 'neutral'])}")
    print(f"   - Note moyenne : {reviews_df['rating'].mean():.2f}")
    
    # Save metadata
    metadata = {
        'generation_date': '2024-01-01',
        'num_reviews': len(reviews_df),
        'num_training_examples': len(training_df),
        'sentiment_distribution': reviews_df['sentiment_type'].value_counts().to_dict(),
        'rating_distribution': reviews_df['rating'].value_counts().to_dict(),
        'categories': reviews_df['product_category'].unique().tolist()
    }
    
    metadata_path = settings.RAW_DATA_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   - Métadonnées : {metadata_path}")
    print("\n✨ Génération terminée avec succès !")

if __name__ == "__main__":
    main()
