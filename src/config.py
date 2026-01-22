"""
Configuration module for the Movie Recommender System.
Centralizes all configuration management using environment variables.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the application."""

    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    MODELS_DIR: Path = OUTPUT_DIR / "models"

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{OUTPUT_DIR}/mlflow.db"
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "movie-recommender")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{OUTPUT_DIR}/recommender.db")

    # Model Configuration
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODELS_DIR)))
    DEFAULT_K_RECOMMENDATIONS: int = int(os.getenv("DEFAULT_K_RECOMMENDATIONS", "10"))

    # Data Configuration
    MOVIELENS_DATASET_SIZE: Literal[
        "latest-small", "1m", "10m", "20m", "25m"
    ] = os.getenv(
        "MOVIELENS_DATASET_SIZE", "latest-small"
    )

    MOVIELENS_URL_MAP: dict[str, str] = {
        "latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    }

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: Literal["json", "text"] = os.getenv("LOG_FORMAT", "text")

    # Feature Flags
    ENABLE_COLD_START_FALLBACK: bool = (
        os.getenv("ENABLE_COLD_START_FALLBACK", "true").lower() == "true"
    )
    ENABLE_POPULARITY_BASELINE: bool = (
        os.getenv("ENABLE_POPULARITY_BASELINE", "true").lower() == "true"
    )

    # Model Hyperparameters
    SVD_N_FACTORS: int = 100
    SVD_N_EPOCHS: int = 20
    SVD_LR_ALL: float = 0.005
    SVD_REG_ALL: float = 0.02

    KNN_K: int = 40
    KNN_MIN_K: int = 1

    # Evaluation
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODELS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
Config.ensure_directories()
