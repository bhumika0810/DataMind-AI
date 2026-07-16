"""
config.py — Load environment variables from .env
"""

import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    """GEMINI_API_KEY: str  = os.getenv("GEMINI_API_KEY", "")"""
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    DATA_DIR: str         = os.getenv(
        "DATA_DIR",
        os.path.join(tempfile.gettempdir(), "datamind_data"),
    )
    UPLOAD_DIR: str       = os.getenv(
        "UPLOAD_DIR",
        os.path.join(tempfile.gettempdir(), "datamind_uploads"),
    )
    DATASET_REGISTRY_PATH: str = os.getenv(
        "DATASET_REGISTRY_PATH",
        os.path.join(DATA_DIR, "datasets.json"),
    )
    MAX_ROWS_FOR_AI: int  = int(os.getenv("MAX_ROWS_FOR_AI", 500))

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


settings = Settings()
