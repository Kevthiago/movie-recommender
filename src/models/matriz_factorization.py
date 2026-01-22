import numpy as np
import pandas as pd
from uvicorn import Config


from typing import Optional, Tuple
from sklearn.decomposition import TruncatedSVD

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)

class MatrixFactorizationRecommender:
    """
    Matrix Factorization Recommender using TruncatedSVD.
    
    This is equivalent to the SVD algorithm from scikit-surprise,
    but implemented using pure scikit-learn (compatible with Python 3.14).
    """

    def __init__(
        self,
        n_factors: int = Config.SVD_N_FACTORS,
        random_state: int = Config.RANDOM_STATE,
    ):
        """
        Initialize the recommender.
        
        Args:
            n_factors: Number of latent factors
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(
            n_components=n_factors,
            random_state=random_state,
            algorithm='randomized'
        )
        
        self.user_ids: Optional[list] = None
        self.movie_ids: Optional[list] = None
        self.user_mean: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        
    def prepare_matrix(
        self, ratings_df: pd.DataFrame
    ) -> Tuple[np.ndarray, list, list, np.ndarray]:
        """
        Convert ratings DataFrame to user-item matrix.
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating columns
            
        Returns:
            Tuple of (matrix, user_ids, movie_ids, user_means)
        """
        logger.info("Creating user-item matrix...")
        
        # Create pivot table
        matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        user_ids = matrix.index.tolist()
        movie_ids = matrix.columns.tolist()
        matrix_values = matrix.values
        
        # Calculate user means for bias correction
        user_means = np.zeros(len(user_ids))
        for i, uid in enumerate(user_ids):
            user_ratings = matrix_values[i][matrix_values[i] > 0]
            if len(user_ratings) > 0:
                user_means[i] = user_ratings.mean()
        
        logger.info(f"Matrix shape: {matrix_values.shape[0]} users × {matrix_values.shape[1]} movies")
        sparsity = 1 - (np.count_nonzero(matrix_values) / matrix_values.size)
        logger.info(f"Sparsity: {sparsity:.2%}")
        
        return matrix_values, user_ids, movie_ids, user_means
    
    def fit(self, ratings_df: pd.DataFrame) -> 'MatrixFactorizationRecommender':
        """
        Fit the model to training data.
        
        Args:
            ratings_df: Training ratings DataFrame
            
        Returns:
            Self (fitted model)
        """
        logger.info("Training Matrix Factorization model...")
        
        # Prepare matrix
        matrix, user_ids, movie_ids, user_means = self.prepare_matrix(ratings_df)
        
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.user_mean = user_means
        self.global_mean = ratings_df['rating'].mean()
        
        # Center the matrix by subtracting user means
        matrix_centered = matrix.copy()
        for i in range(len(user_ids)):
            mask = matrix[i] > 0
            if mask.any():
                matrix_centered[i][mask] -= user_means[i]
        
        # Fit SVD
        logger.info(f"Fitting SVD with {self.n_factors} factors...")
        self.user_factors = self.svd.fit_transform(matrix_centered)
        self.item_factors = self.svd.components_.T
        
        # Log variance explained
        variance_explained = self.svd.explained_variance_ratio_.sum()
        logger.info(f"Variance explained: {variance_explained:.2%}")
        
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (clipped to 0.5-5.0 range)
        """
        try:
            user_idx = self.user_ids.index(user_id)
            movie_idx = self.movie_ids.index(movie_id)
            
            # Reconstruct rating: user_mean + user_factor · item_factor
            prediction = (
                self.user_mean[user_idx] +
                np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
            )
            
            # Clip to valid rating range
            return np.clip(prediction, 0.5, 5.0)
            
        except ValueError:
            # User or movie not in training set - return global mean
            return self.global_mean
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True
    ) -> list[tuple[int, float]]:
        """
        Get top-K recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        try:
            user_idx = self.user_ids.index(user_id)
        except ValueError:
            # Cold start - return most popular movies
            logger.warning(f"User {user_id} not in training set (cold start)")
            return []
        
        # Predict for all movies
        predictions = []
        for movie_idx, movie_id in enumerate(self.movie_ids):
            pred_rating = (
                self.user_mean[user_idx] +
                np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
            )
            predictions.append((movie_id, np.clip(pred_rating, 0.5, 5.0)))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:k]