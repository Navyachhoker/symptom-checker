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

    @property
    def cors_origins(self) -> list[str]:
        """
        Returns a list of allowed CORS origins parsed from frontend_url.
        Supports a single URL or a comma-separated list
        (e.g. "https://app.example.com,https://staging.example.com").
        Falls back to ["*"] only if frontend_url was left at its default.
        """
        if not self.frontend_url or self.frontend_url == "*":
            return ["*"]
        return [origin.strip() for origin in self.frontend_url.split(",") if origin.strip()]

settings = Settings()