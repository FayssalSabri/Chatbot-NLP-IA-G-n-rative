# 🤖 Chatbot NLP & IA Générative

Assistant intelligent combinant recherche sémantique et génération de réponses pour l'analyse d'avis clients avec application de règles métier.

## 🚀 Fonctionnalités

- **🔍 Recherche sémantique** avec FAISS et embeddings
- **🎭 Analyse de sentiment** et extraction d'aspects
- **⚡ Règles métier automatiques** pour le traitement des avis
- **🤖 Fine-tuning GPT-4** pour des réponses personnalisées
- **📊 Interface Chainlit** intuitive
- **🔌 API REST Flask** pour intégration
- **🐳 Déploiement Docker** complet

## 📋 Prérequis

- Python 3.11+
- Docker & Docker Compose
- Clé API OpenAI
- 8GB RAM minimum (pour FAISS)

## ⚡ Installation Rapide

1. **Cloner le repository**
```bash
git clone <repository-url>
cd chatbot-nlp-project
```

2. **Configuration**
```bash
# Copier le fichier d'environnement
cp .env.template .env

# Éditer avec vos clés API
nano .env
```

3. **Installation des dépendances**
```bash
make setup-dev
```

4. **Génération de données d'exemple**
```bash
make sample-data
```

5. **Lancement avec Docker**
```bash
make build
make up
```

## 🎯 Utilisation

### Interface Chainlit (Recommandée)
```
http://localhost:8000
```

### API REST Flask
```
http://localhost:5000
```

## 📚 Documentation API

### Endpoints principaux

#### POST /analyze
Analyse complète d'un avis client
```json
{
    "text": "Le service était décevant mais le produit correct",
    "rating": 3
}
```

#### POST /search
Recherche sémantique d'avis similaires
```json
{
    "query": "problème de qualité",
    "k": 10,
    "filters": {"category": "Electronics"}
}
```

#### POST /preprocess
Préprocessing de texte
```json
{
    "text": "Avis client à analyser"
}
```

## 🔧 Architecture

```
├── src/
│   ├── data/              # Préprocessing et embeddings
│   ├── models/            # FAISS, GPT fine-tuning
│   ├── business_rules/    # Moteur de règles métier
│   ├── api/              # Flask et Chainlit
│   └── utils/            # Configuration
├── data/                 # Données brutes et traitées
├── models/              # Modèles entraînés et index
└── docker/             # Configuration Docker
```

## 🧠 Composants Techniques

### 1. Preprocessing (NLP)
- **spaCy** : Extraction d'entités
- **TextBlob** : Analyse de sentiment
- **Nettoyage avancé** : Regex et normalisation

### 2. Embeddings & Recherche
- **Sentence Transformers** : Génération d'embeddings
- **FAISS** : Index vectoriel haute performance
- **Recherche sémantique** avec filtres

### 3. Règles Métier
- **Moteur de règles** flexible et configurable
- **Actions automatiques** : escalade, catégorisation
- **Priorités** et conditions personnalisables

### 4. IA Générative
- **Fine-tuning GPT-4** sur données métier
- **Génération contextuelle** de réponses
- **Templates adaptatifs**

## ⚙️ Configuration Avancée

### Variables d'environnement
```bash
# API Keys
OPENAI_API_KEY=your_key_here

# Modèles
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GPT_MODEL=gpt-4
FINE_TUNED_MODEL=ft:gpt-4:custom

# Base de données
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379
```

### Règles métier personnalisées
```python
# Exemple de règle personnalisée
rule = BusinessRule(
    name="vip_customer_complaint",
    description="Traitement prioritaire VIP",
    conditions=[
        {"type": "rating", "threshold": 3, "operator": "lt"},
        {"type": "keyword", "keywords": ["vip", "premium"], "match_any": True}
    ],
    actions=[
        {"type": "escalate", "department": "vip_support", "priority": "critical"},
        {"type": "send_email", "recipient": "vip-manager@company.com"}
    ],
    priority=Priority.CRITICAL
)

rules_engine.add_rule(rule)
```

## 📊 Monitoring

### Métriques disponibles
- **Performance recherche** : temps de réponse FAISS
- **Qualité modèle** : scores de confiance GPT
- **Règles métier** : taux de déclenchement
- **Satisfaction utilisateur** : feedback interface

### Logs structurés
```python
# Configuration logging
import structlog
logger = structlog.get_logger()
```

## 🧪 Tests

```bash
# Tests unitaires
make test

# Tests d'intégration
python -m pytest tests/integration/

# Tests de performance
python scripts/benchmark_faiss.py
```

## 🚀 Déploiement Production

### Docker Compose Production
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    image: chatbot-nlp:latest
    environment:
      - FLASK_ENV=production
      - WORKERS=4
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### Scaling FAISS
```python
# Configuration haute performance
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128
```

## 🔧 Développement

### Ajout de nouvelles fonctionnalités

1. **Nouveau type de condition** :
```python
class CustomCondition(Condition):
    def evaluate(self, context):
        # Logique personnalisée
        return True
```

2. **Extension API** :
```python
@app.route('/custom-endpoint')
def custom_feature():
    # Nouvelle fonctionnalité
    pass
```

### Fine-tuning personnalisé
```python
# Préparation données
df = prepare_custom_training_data()
fine_tuner = GPTFineTuner(api_key)
file_id = fine_tuner.upload_training_file("training.jsonl")
job_id = fine_tuner.create_fine_tune_job(file_id)
model_id = fine_tuner.wait_for_completion(job_id)
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🆘 Support

- **Documentation** : [Wiki du projet](link-to-wiki)
- **Issues** : [GitHub Issues](link-to-issues)
- **Discord** : [Communauté développeurs](link-to-discord)

## 🎯 Roadmap

- [ ] **v1.1** : Support multi-langues
- [ ] **v1.2** : Intégration Elasticsearch
- [ ] **v1.3** : Tableau de bord analytique
- [ ] **v2.0** : Architecture microservices
```

## 17. docker/nginx.conf
```nginx
upstream flask_app {
    server chatbot-api:5000;
}

upstream chainlit_app {
    server chainlit-ui:8000;
}

server {
    listen 80;
    server_name localhost;

    # Logs
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # API routes
    location /api/ {
        proxy_pass http://flask_app/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Chainlit UI
    location / {
        proxy_pass http://chainlit_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Chainlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check
    location /health {
        proxy_pass http://flask_app/health;
    }

    # Static files caching
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```
