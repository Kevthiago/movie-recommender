"""
Model training module for the Movie Recommender System.
Implements collaborative filtering using ONLY scikit-learn (Python 3.14 compatible).
No external compilation required - works out of the box!
"""

from src.models.matriz_factorization import MatrixFactorizationRecommender

import joblib
from pathlib import Path
from typing import Any, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import NearestNeighbors

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


class RecommenderTrainer:
    """Train and evaluate recommendation models."""

    def __init__(self, experiment_name: str = Config.MLFLOW_EXPERIMENT_NAME):
        """
        Initialize the trainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        self.model: Optional[MatrixFactorizationRecommender] = None

    def prepare_data(
        self, ratings: pd.DataFrame, test_size: float = Config.TEST_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            ratings: Full ratings DataFrame
            test_size: Proportion for test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Splitting data into train/test sets...")
        
        train_df, test_df = train_test_split(
            ratings,
            test_size=test_size,
            random_state=Config.RANDOM_STATE
        )
        
        logger.info(f"Training set: {len(train_df):,} ratings")
        logger.info(f"Test set: {len(test_df):,} ratings")
        
        return train_df, test_df

    def train_model(
        self,
        train_df: pd.DataFrame,
        n_factors: int = Config.SVD_N_FACTORS,
    ) -> MatrixFactorizationRecommender:
        """
        Train Matrix Factorization model.
        
        Args:
            train_df: Training DataFrame
            n_factors: Number of latent factors
            
        Returns:
            Trained model
        """
        logger.info("Training Matrix Factorization model...")

        with mlflow.start_run(run_name="MatrixFactorization_Model"):
            # Log parameters
            params = {
                "algorithm": "TruncatedSVD",
                "n_factors": n_factors,
                "python_version": "3.14",
                "framework": "scikit-learn",
            }
            mlflow.log_params(params)

            # Train model
            model = MatrixFactorizationRecommender(n_factors=n_factors)
            model.fit(train_df)

            self.model = model
            
            # Log model info
            mlflow.log_metric("n_users", len(model.user_ids))
            mlflow.log_metric("n_items", len(model.movie_ids))

        logger.info("Model training complete")
        return model

    def evaluate_on_test(
        self, model: MatrixFactorizationRecommender, test_df: pd.DataFrame
    ) -> dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating on test set...")

        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            pred = model.predict(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        metrics = {"test_rmse": float(rmse), "test_mae": float(mae)}

        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")

        return metrics

    def save_model(
        self,
        model: MatrixFactorizationRecommender,
        name: str = "svd_model",
        metadata: Optional[dict] = None
    ) -> Path:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            name: Model filename
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        model_path = Config.MODELS_DIR / f"{name}.pkl"

        logger.info(f"Saving model to {model_path}")

        with open(model_path, "wb") as f:
            joblib.dump(
                {
                    "model": model,
                    "metadata": metadata or {},
                    "config": {
                        "n_factors": model.n_factors,
                        "python_version": "3.14",
                    },
                },
                f,
            )

        # Log to MLflow
        mlflow.log_artifact(str(model_path))

        logger.info("Model saved successfully")
        return model_path

    @staticmethod
    def load_model(model_path: Path) -> dict[str, Any]:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary containing model and metadata
        """
        logger.info(f"Loading model from {model_path}")

        with open(model_path, "rb") as f:
            model_data = joblib.load(f)

        logger.info("Model loaded successfully")
        return model_data


def main() -> None:
    """Main training pipeline."""
    logger.info("=== Starting Model Training Pipeline (Python 3.14) ===")

    # Load processed data
    logger.info("Loading processed data...")
    features_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "features.parquet")

    # Initialize trainer
    trainer = RecommenderTrainer()

    # Prepare data
    train_df, test_df = trainer.prepare_data(
        features_df[["userId", "movieId", "rating"]]
    )

    # Train model
    model = trainer.train_model(train_df)

    # Evaluate
    test_metrics = trainer.evaluate_on_test(model, test_df)

    # Log metrics to MLflow
    with mlflow.start_run(run_name="Evaluation"):
        mlflow.log_metrics(test_metrics)

    # Save model
    metadata = {
        "n_users": len(model.user_ids),
        "n_items": len(model.movie_ids),
        **test_metrics,
    }
    trainer.save_model(model, name="svd_recommender", metadata=metadata)

    logger.info("\n=== Training Complete ===")
    logger.info(f"Model saved to: {Config.MODELS_DIR}")
    logger.info(f"MLflow tracking: {Config.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
