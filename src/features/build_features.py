"""
Feature engineering module for the Movie Recommender System.
Transforms raw data into features suitable for machine learning models.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class FeatureBuilder:
    """Build features from raw MovieLens data."""

    def __init__(self) -> None:
        """Initialize the feature builder."""
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.is_fitted = False

    def fit_transform(
        self, ratings: pd.DataFrame, movies: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Fit encoders and transform data.

        Args:
            ratings: Raw ratings DataFrame
            movies: Raw movies DataFrame

        Returns:
            Tuple of (transformed_data, metadata)
        """
        logger.info("Starting feature engineering...")

        # Create a copy to avoid modifying original
        df = ratings.copy()

        # Encode user and movie IDs to contiguous integers
        logger.info("Encoding user and movie IDs...")
        df["user_idx"] = self.user_encoder.fit_transform(df["userId"])
        df["movie_idx"] = self.movie_encoder.fit_transform(df["movieId"])

        # Add movie metadata
        logger.info("Merging movie metadata...")
        df = df.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")

        # Parse genres
        logger.info("Processing genres...")
        df["genres_list"] = df["genres"].apply(lambda x: x.split("|") if pd.notna(x) else [])
        df["n_genres"] = df["genres_list"].apply(len)

        # Temporal features
        logger.info("Creating temporal features...")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour"] = df["timestamp"].dt.hour

        # User features
        logger.info("Computing user features...")
        user_stats = df.groupby("userId").agg(
            {
                "rating": ["mean", "std", "count"],
                "timestamp": ["min", "max"],
            }
        )
        user_stats.columns = ["_".join(col).strip() for col in user_stats.columns.values]
        user_stats = user_stats.reset_index()
        user_stats.columns = [
            "userId",
            "user_avg_rating",
            "user_rating_std",
            "user_n_ratings",
            "user_first_rating",
            "user_last_rating",
        ]

        df = df.merge(user_stats, on="userId", how="left")

        # Movie features
        logger.info("Computing movie features...")
        movie_stats = df.groupby("movieId").agg(
            {
                "rating": ["mean", "std", "count"],
                "timestamp": ["min", "max"],
            }
        )
        movie_stats.columns = ["_".join(col).strip() for col in movie_stats.columns.values]
        movie_stats = movie_stats.reset_index()
        movie_stats.columns = [
            "movieId",
            "movie_avg_rating",
            "movie_rating_std",
            "movie_n_ratings",
            "movie_first_rating",
            "movie_last_rating",
        ]

        df = df.merge(movie_stats, on="movieId", how="left")

        # Popularity score (weighted rating)
        logger.info("Computing popularity scores...")
        C = df["movie_avg_rating"].mean()  # Mean rating across all movies
        m = df["movie_n_ratings"].quantile(0.75)  # Minimum votes required

        df["popularity_score"] = (
            df["movie_n_ratings"] / (df["movie_n_ratings"] + m) * df["movie_avg_rating"]
            + m / (df["movie_n_ratings"] + m) * C
        )

        # Sparsity metrics
        n_users = df["userId"].nunique()
        n_movies = df["movieId"].nunique()
        n_ratings = len(df)
        sparsity = 1 - (n_ratings / (n_users * n_movies))

        metadata = {
            "n_users": n_users,
            "n_movies": n_movies,
            "n_ratings": n_ratings,
            "sparsity": sparsity,
            "rating_mean": df["rating"].mean(),
            "rating_std": df["rating"].std(),
            "user_encoder": self.user_encoder,
            "movie_encoder": self.movie_encoder,
        }

        self.is_fitted = True

        logger.info("Feature engineering complete")
        logger.info(f"  Users: {n_users:,}")
        logger.info(f"  Movies: {n_movies:,}")
        logger.info(f"  Ratings: {n_ratings:,}")
        logger.info(f"  Sparsity: {sparsity:.2%}")

        return df, metadata

    def get_top_popular_movies(
        self, df: pd.DataFrame, n: int = 100
    ) -> pd.DataFrame:
        """
        Get the top N most popular movies.

        Args:
            df: Processed ratings DataFrame
            n: Number of movies to return

        Returns:
            DataFrame with top popular movies
        """
        popular = (
            df.groupby(["movieId", "title"])
            .agg(
                {
                    "rating": "mean",
                    "movie_n_ratings": "first",
                    "popularity_score": "first",
                }
            )
            .reset_index()
        )

        popular = popular.sort_values("popularity_score", ascending=False).head(n)

        logger.info(f"Retrieved top {n} popular movies")
        return popular

    def save_features(self, df: pd.DataFrame, metadata: dict[str, Any]) -> None:
        """
        Save engineered features and metadata.

        Args:
            df: Processed DataFrame
            metadata: Feature metadata
        """
        features_path = Config.PROCESSED_DATA_DIR / "features.parquet"
        logger.info(f"Saving features to {features_path}")
        df.to_parquet(features_path, index=False)

        # Save encoders for later use
        import joblib

        encoders_path = Config.PROCESSED_DATA_DIR / "encoders.joblib"
        logger.info(f"Saving encoders to {encoders_path}")
        joblib.dump(
            {
                "user_encoder": self.user_encoder,
                "movie_encoder": self.movie_encoder,
                "metadata": metadata,
            },
            encoders_path,
        )

        logger.info("Features and metadata saved successfully")


def main() -> None:
    """Main function to build features."""
    from src.data.loader import MovieLensLoader

    # Load data
    loader = MovieLensLoader()
    ratings = loader.load_ratings(processed=True)
    movies = loader.load_movies(processed=True)

    # Build features
    builder = FeatureBuilder()
    features_df, metadata = builder.fit_transform(ratings, movies)

    # Save
    builder.save_features(features_df, metadata)

    # Display sample
    logger.info("\n=== Feature Sample ===")
    logger.info(f"\n{features_df.head()}")

    # Get popular movies
    popular = builder.get_top_popular_movies(features_df, n=10)
    logger.info("\n=== Top 10 Popular Movies ===")
    logger.info(f"\n{popular[['title', 'rating', 'movie_n_ratings', 'popularity_score']]}")


if __name__ == "__main__":
    main()
