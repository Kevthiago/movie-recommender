"""
Prediction module for the Movie Recommender System.
Generates personalized recommendations using scikit-learn models (Python 3.14 compatible).
"""

from src.models.matriz_factorization import MatrixFactorizationRecommender

import joblib
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.logger import setup_logger

from src.models.matriz_factorization import MatrixFactorizationRecommender

logger = setup_logger(__name__)


class MovieRecommender:
    """Generate movie recommendations using trained models."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the recommender.
        
        Args:
            model_path: Path to the trained model file
        """
        if model_path is None:
            model_path = Config.MODELS_DIR / "svd_recommender.pkl"

        self.model_path = model_path
        self.model: Optional[Any] = None
        self.metadata: dict[str, Any] = {}
        self.movies_df: Optional[pd.DataFrame] = None
        self.popular_movies: Optional[pd.DataFrame] = None

        self._load_model()
        self._load_data()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        logger.info(f"Loading model from {self.model_path}")

        try:
            import sys
            sys.modules['__main__'].MatrixFactorizationRecommender = MatrixFactorizationRecommender
            
            with open(self.model_path, "rb") as f:
                model_data = joblib.load(f)

            self.model = model_data["model"]
            self.metadata = model_data.get("metadata", {})

            logger.info("Model loaded successfully")
            logger.info(f"Model metadata: {self.metadata}")

        except FileNotFoundError:
            logger.error(f"Model not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_data(self) -> None:
        """Load supporting data (movies, encoders)."""
        logger.info("Loading supporting data...")

        # Load movies
        movies_path = Config.PROCESSED_DATA_DIR / "movies.parquet"
        self.movies_df = pd.read_parquet(movies_path)

        # Prepare popular movies for cold start
        features_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "features.parquet")
        self.popular_movies = (
            features_df.groupby(["movieId", "title"])
            .agg({"popularity_score": "first", "movie_avg_rating": "first"})
            .reset_index()
            .sort_values("popularity_score", ascending=False)
        )

        logger.info(f"Loaded {len(self.movies_df)} movies")

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        prediction = self.model.predict(user_id, movie_id)
        return float(prediction)

    def recommend_for_user(
        self,
        user_id: int,
        k: int = Config.DEFAULT_K_RECOMMENDATIONS,
        exclude_rated: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Generate top-K movie recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations to generate
            exclude_rated: If True, exclude movies the user has already rated
            
        Returns:
            List of recommendation dictionaries
        """
        logger.info(f"Generating {k} recommendations for user {user_id}")

        if self.model is None or self.movies_df is None:
            raise RuntimeError("Model or data not loaded")

        try:
            # Check if user exists in training set
            if user_id not in self.model.user_ids:
                logger.warning(f"User {user_id} not in training set - using popular movies")
                return self._get_popular_recommendations(k)

            # Get recommendations from model
            movie_predictions = self.model.recommend(
                user_id=user_id,
                k=k * 2,  # Get more to account for filtering
                exclude_seen=exclude_rated
            )

            if not movie_predictions:
                return self._get_popular_recommendations(k)

            # Exclude already rated movies if requested
            if exclude_rated:
                features_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "features.parquet")
                rated_movies = set(
                    features_df[features_df["userId"] == user_id]["movieId"].values
                )
                movie_predictions = [
                    (mid, rating) for mid, rating in movie_predictions
                    if mid not in rated_movies
                ]

            # Take top K
            movie_predictions = movie_predictions[:k]

            # Build recommendations with movie details
            recommendations = []
            for movie_id, pred_rating in movie_predictions:
                movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
                
                if movie_info.empty:
                    continue
                    
                movie_info = movie_info.iloc[0]
                recommendations.append(
                    {
                        "movie_id": int(movie_id),
                        "title": movie_info["title"],
                        "genres": movie_info["genres"],
                        "predicted_rating": round(float(pred_rating), 2),
                    }
                )

            # If we don't have enough, supplement with popular
            if len(recommendations) < k:
                logger.warning(f"Only found {len(recommendations)} recommendations, supplementing with popular movies")
                popular = self._get_popular_recommendations(k - len(recommendations))
                recommendations.extend(popular)

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations[:k]

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Fall back to popular movies
            return self._get_popular_recommendations(k)

    def _get_popular_recommendations(self, k: int) -> list[dict[str, Any]]:
        """
        Get popular movies as fallback (cold start strategy).
        
        Args:
            k: Number of recommendations
            
        Returns:
            List of popular movie recommendations
        """
        logger.info(f"Using popular movies as fallback (cold start)")

        if self.popular_movies is None:
            raise RuntimeError("Popular movies not loaded")

        top_popular = self.popular_movies.head(k)

        recommendations = []
        for _, row in top_popular.iterrows():
            movie_info = self.movies_df[self.movies_df["movieId"] == row["movieId"]]
            
            if movie_info.empty:
                continue
                
            movie_info = movie_info.iloc[0]
            recommendations.append(
                {
                    "movie_id": int(row["movieId"]),
                    "title": row["title"],
                    "genres": movie_info["genres"],
                    "predicted_rating": round(float(row.get("movie_avg_rating", 0)), 2),
                    "reason": "popular",
                }
            )

        return recommendations

    def get_similar_movies(
        self, movie_id: int, k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Find similar movies based on latent factors.
        
        Args:
            movie_id: Reference movie ID
            k: Number of similar movies to return
            
        Returns:
            List of similar movie recommendations
        """
        logger.info(f"Finding {k} movies similar to movie {movie_id}")

        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Get movie index
            movie_idx = self.model.movie_ids.index(movie_id)
            movie_factors = self.model.item_factors[movie_idx]

            # Compute similarity with all other movies
            similarities = []
            for idx, other_movie_id in enumerate(self.model.movie_ids):
                if idx == movie_idx:
                    continue

                other_factors = self.model.item_factors[idx]
                
                # Cosine similarity
                similarity = np.dot(movie_factors, other_factors) / (
                    np.linalg.norm(movie_factors) * np.linalg.norm(other_factors)
                )

                similarities.append((other_movie_id, float(similarity)))

            # Sort by similarity and get top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:k]

            # Build recommendations
            recommendations = []
            for similar_movie_id, similarity_score in top_k:
                movie_info = self.movies_df[
                    self.movies_df["movieId"] == similar_movie_id
                ]
                
                if movie_info.empty:
                    continue
                    
                movie_info = movie_info.iloc[0]
                recommendations.append(
                    {
                        "movie_id": int(similar_movie_id),
                        "title": movie_info["title"],
                        "genres": movie_info["genres"],
                        "similarity_score": round(similarity_score, 3),
                    }
                )

            logger.info(f"Found {len(recommendations)} similar movies")
            return recommendations

        except ValueError as e:
            logger.error(f"Movie {movie_id} not found in training set: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []


def main() -> None:
    """Demo of the recommendation system."""
    logger.info("=== Movie Recommender Demo (Python 3.14) ===")

    # Initialize recommender
    recommender = MovieRecommender()

    # Example: Recommend for user 1
    user_id = 1
    recommendations = recommender.recommend_for_user(user_id, k=10)

    logger.info(f"\nTop 10 recommendations for User {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(
            f"{i}. {rec['title']} (Predicted: {rec.get('predicted_rating', 'N/A')}, "
            f"Genres: {rec['genres']})"
        )

    # Example: Find similar movies
    if recommender.model and recommender.model.movie_ids:
        movie_id = recommender.model.movie_ids[0]  # First movie in training set
        similar = recommender.get_similar_movies(movie_id, k=5)

        logger.info(f"\nMovies similar to movie {movie_id}:")
        for i, sim in enumerate(similar, 1):
            logger.info(
                f"{i}. {sim['title']} (Similarity: {sim['similarity_score']}, "
                f"Genres: {sim['genres']})"
            )


if __name__ == "__main__":
    main()
