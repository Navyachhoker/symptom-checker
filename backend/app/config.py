from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

class Settings(BaseSettings):
    database_url: str
    groq_api_key: str
    app_env: str = "development"
    secret_key: str = "change-me"
    frontend_url: str = "*"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-fix Render's postgres:// URL for asyncpg
        if self.database_url.startswith("postgresql://"):
            object.__setattr__(
                self, "database_url",
                self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            )

    class Config:
        env_file = "../../.env"
        extra = "ignore"

settings = Settings()