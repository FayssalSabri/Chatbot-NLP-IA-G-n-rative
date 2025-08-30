import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = str(MODELS_DIR / "faiss_index")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # GPT Configuration
    GPT_MODEL: str = "gpt-4"
    FINE_TUNED_MODEL: str = ""
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    
    # Database
    POSTGRES_URL: str = os.getenv("DATABASE_URL", "postgresql://chatbot_user:chatbot_pass@localhost:5432/chatbot_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Flask
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create directories if they don't exist
for directory in [settings.DATA_DIR, settings.MODELS_DIR, 
                 settings.RAW_DATA_DIR, settings.PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
