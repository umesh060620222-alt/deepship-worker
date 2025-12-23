import os
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/deepsearch")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    # Scraper API
    SCRAPER_API_URL: str = os.getenv("SCRAPER_API_URL", "http://localhost:8001")
    
    # LLM API (adjust based on your provider)
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_API_URL: str = os.getenv("LLM_API_URL", "https://api.openai.com/v1")
    
    # Task settings
    TASK_SOFT_TIME_LIMIT: int = 180  # 3 minutes
    TASK_TIME_LIMIT: int = 200
    TASK_MAX_RETRIES: int = 2
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Worker settings
    CELERY_WORKER_CONCURRENCY_LLM: int = int(os.getenv("CELERY_WORKER_CONCURRENCY_LLM", "10"))
    CELERY_WORKER_CONCURRENCY_SCRAPER: int = int(os.getenv("CELERY_WORKER_CONCURRENCY_SCRAPER", "2"))

config = Config()
