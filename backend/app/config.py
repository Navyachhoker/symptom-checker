from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load .env from project root (one level up from backend/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

class Settings(BaseSettings):
    database_url: str
    groq_api_key: str
    app_env: str = "development"
    secret_key: str = "change-me"

    class Config:
        env_file = "../../.env"
        extra = "ignore"

settings = Settings()