"""
Unit tests for the data loading module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.loader import MovieLensLoader


class TestMovieLensLoader:
    """Test suite for MovieLensLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a MovieLensLoader instance."""
        return MovieLensLoader(dataset_size="small")

    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader.dataset_size == "small"
        assert loader.download_url is not None
        assert "grouplens.org" in loader.download_url

    def test_invalid_dataset_size(self):
        """Test that invalid dataset size raises error."""
        with pytest.raises(ValueError):
            MovieLensLoader(dataset_size="invalid_size")

    def test_raw_data_directory_exists(self, loader):
        """Test that raw data directory exists."""
        assert loader.raw_data_dir.exists()
        assert loader.raw_data_dir == Config.RAW_DATA_DIR

    @pytest.mark.skipif(
        not (Config.RAW_DATA_DIR / "ml-small.zip").exists(),
        reason="Dataset not downloaded",
    )
    def test_load_ratings(self, loader):
        """Test loading ratings data."""
        ratings = loader.load_ratings()

        # Check DataFrame structure
        assert isinstance(ratings, pd.DataFrame)
        assert "userId" in ratings.columns
        assert "movieId" in ratings.columns
        assert "rating" in ratings.columns
        assert "timestamp" in ratings.columns

        # Check data types
        assert pd.api.types.is_integer_dtype(ratings["userId"])
        assert pd.api.types.is_integer_dtype(ratings["movieId"])
        assert pd.api.types.is_float_dtype(ratings["rating"])

        # Check rating range
        assert ratings["rating"].min() >= 0.5
        assert ratings["rating"].max() <= 5.0

    @pytest.mark.skipif(
        not (Config.RAW_DATA_DIR / "ml-small.zip").exists(),
        reason="Dataset not downloaded",
    )
    def test_load_movies(self, loader):
        """Test loading movies data."""
        movies = loader.load_movies()

        # Check DataFrame structure
        assert isinstance(movies, pd.DataFrame)
        assert "movieId" in movies.columns
        assert "title" in movies.columns
        assert "genres" in movies.columns

        # Check data types
        assert pd.api.types.is_integer_dtype(movies["movieId"])
        assert pd.api.types.is_string_dtype(movies["title"])

    def test_config_directories_exist(self):
        """Test that all configuration directories are created."""
        assert Config.RAW_DATA_DIR.exists()
        assert Config.PROCESSED_DATA_DIR.exists()
        assert Config.OUTPUT_DIR.exists()
        assert Config.MODELS_DIR.exists()


class TestConfig:
    """Test suite for Config class."""

    def test_config_paths(self):
        """Test that config paths are correctly set."""
        assert Config.BASE_DIR.exists()
        assert Config.DATA_DIR.exists()

    def test_config_api_settings(self):
        """Test API configuration settings."""
        assert isinstance(Config.API_HOST, str)
        assert isinstance(Config.API_PORT, int)
        assert Config.API_PORT > 0

    def test_config_model_settings(self):
        """Test model configuration settings."""
        assert Config.SVD_N_FACTORS > 0
        assert Config.SVD_N_EPOCHS > 0
        assert 0 < Config.TEST_SIZE < 1

    def test_config_movielens_urls(self):
        """Test that MovieLens URLs are properly configured."""
        assert "small" in Config.MOVIELENS_URL_MAP
        assert "1m" in Config.MOVIELENS_URL_MAP
        assert all(
            "grouplens.org" in url for url in Config.MOVIELENS_URL_MAP.values()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
