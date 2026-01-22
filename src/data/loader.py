"""
Data loading module for the Movie Recommender System.
Handles downloading, extracting, and loading the MovieLens dataset.
"""

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class MovieLensLoader:
    """Loader for MovieLens datasets."""

    def __init__(self, dataset_size: str = Config.MOVIELENS_DATASET_SIZE):
        """
        Initialize the MovieLens loader.

        Args:
            dataset_size: Size of the dataset to load ('small', '1m', '10m', '20m', '25m')
        """
        self.dataset_size = dataset_size
        self.download_url = Config.MOVIELENS_URL_MAP.get(dataset_size)
        if not self.download_url:
            raise ValueError(f"Invalid dataset size: {dataset_size}")

        self.raw_data_dir = Config.RAW_DATA_DIR
        self.zip_path = self.raw_data_dir / f"ml-{dataset_size}.zip"
        self.extract_dir = self.raw_data_dir / f"ml-{dataset_size}"

    def download_dataset(self, force: bool = False) -> Path:
        """
        Download the MovieLens dataset.

        Args:
            force: If True, re-download even if file exists

        Returns:
            Path to the downloaded zip file
        """
        if self.zip_path.exists() and not force:
            logger.info(f"Dataset already exists at {self.zip_path}")
            return self.zip_path

        logger.info(f"Downloading MovieLens {self.dataset_size} dataset...")

        try:
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(self.zip_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Dataset downloaded successfully to {self.zip_path}")
            return self.zip_path

        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def extract_dataset(self) -> Path:
        """
        Extract the downloaded dataset.

        Returns:
            Path to the extracted directory
        """
        extract_dir = self.raw_data_dir / f"ml-{self.dataset_size}"

        if self.extract_dir.exists():
            logger.info(f"Dataset already extracted at {self.extract_dir}")
            return self.extract_dir

        logger.info(f"Extracting dataset to {self.extract_dir}...")

        try:
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_data_dir)

            logger.info("Dataset extracted successfully")
            return extract_dir

        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract dataset: {e}")
            raise

    def load_ratings(self, processed: bool = False) -> pd.DataFrame:
        """
        Load the ratings dataset.

        Args:
            processed: If True, load from processed directory; otherwise from raw

        Returns:
            DataFrame containing ratings data
        """
        if processed:
            ratings_path = Config.PROCESSED_DATA_DIR / "ratings.parquet"
            logger.info(f"Loading processed ratings from {ratings_path}")
            return pd.read_parquet(ratings_path)

        extract_dir = self.extract_dir

        if self.dataset_size == "1m":
            ratings_path = extract_dir / "ratings.dat"
            logger.info(f"Loading ratings from {ratings_path}")
            ratings = pd.read_csv(
                ratings_path,
                sep="::",
                engine="python",
                names=["userId", "movieId", "rating", "timestamp"],
                encoding="latin-1",
            )
        else:
            ratings_path = extract_dir / "ratings.csv"
            logger.info(f"Loading ratings from {ratings_path}")
            ratings = pd.read_csv(ratings_path)

        logger.info(f"Loaded {len(ratings):,} ratings")
        return ratings

    def load_movies(self, processed: bool = False) -> pd.DataFrame:
        """
        Load the movies dataset.

        Args:
            processed: If True, load from processed directory; otherwise from raw

        Returns:
            DataFrame containing movie metadata
        """
        if processed:
            movies_path = Config.PROCESSED_DATA_DIR / "movies.parquet"
            logger.info(f"Loading processed movies from {movies_path}")
            return pd.read_parquet(movies_path)

        extract_dir = self.extract_dir

        # Handle different file formats
        if self.dataset_size in ["1m"]:
            movies_path = extract_dir / "movies.dat"
            logger.info(f"Loading movies from {movies_path}")
            movies = pd.read_csv(
                movies_path,
                sep="::",
                engine="python",
                names=["movieId", "title", "genres"],
                encoding="latin-1",
            )
        else:
            movies_path = extract_dir / "movies.csv"
            logger.info(f"Loading movies from {movies_path}")
            movies = pd.read_csv(movies_path)

        logger.info(f"Loaded {len(movies):,} movies")
        return movies

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download, extract, and load all datasets.

        Returns:
            Tuple of (ratings, movies) DataFrames
        """
        self.download_dataset()
        self.extract_dataset()

        ratings = self.load_ratings()
        movies = self.load_movies()

        return ratings, movies

    def save_processed(
        self, ratings: pd.DataFrame, movies: pd.DataFrame
    ) -> tuple[Path, Path]:
        """
        Save processed datasets in Parquet format for faster loading.

        Args:
            ratings: Ratings DataFrame
            movies: Movies DataFrame

        Returns:
            Tuple of paths to saved files
        """
        ratings_path = Config.PROCESSED_DATA_DIR / "ratings.parquet"
        movies_path = Config.PROCESSED_DATA_DIR / "movies.parquet"

        logger.info(f"Saving processed ratings to {ratings_path}")
        ratings.to_parquet(ratings_path, index=False)

        logger.info(f"Saving processed movies to {movies_path}")
        movies.to_parquet(movies_path, index=False)

        logger.info("Processed data saved successfully")
        return ratings_path, movies_path


def main() -> None:
    """Main function to download and prepare the dataset."""
    loader = MovieLensLoader()

    logger.info("Starting data download and preparation...")
    ratings, movies = loader.load_all()

    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Ratings: {len(ratings):,} records")
    logger.info(f"Movies: {len(movies):,} records")
    logger.info(f"Users: {ratings['userId'].nunique():,}")
    logger.info(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")

    # Save in optimized format
    loader.save_processed(ratings, movies)

    logger.info("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
