"""
FastAPI application for the Movie Recommender System.
Provides REST endpoints for generating movie recommendations.
"""

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import Config
from src.logger import setup_logger
from src.models.predict import MovieRecommender

from src.models.matriz_factorization import MatrixFactorizationRecommender

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="Production-ready movie recommendation system with collaborative filtering",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender: Optional[MovieRecommender] = None


# Pydantic models for request/response
class RecommendationResponse(BaseModel):
    """Response model for movie recommendations."""

    movie_id: int = Field(..., description="Movie identifier")
    title: str = Field(..., description="Movie title")
    genres: str = Field(..., description="Movie genres (pipe-separated)")
    predicted_rating: Optional[float] = Field(None, description="Predicted rating (0.5-5.0)")
    similarity_score: Optional[float] = Field(None, description="Similarity score (for similar movies)")
    reason: Optional[str] = Field(None, description="Reason for recommendation")


class UserRecommendationsResponse(BaseModel):
    """Response model for user-specific recommendations."""

    user_id: int = Field(..., description="User identifier")
    k: int = Field(..., description="Number of recommendations")
    recommendations: list[RecommendationResponse]


class SimilarMoviesResponse(BaseModel):
    """Response model for similar movies."""

    movie_id: int = Field(..., description="Reference movie identifier")
    k: int = Field(..., description="Number of similar movies")
    similar_movies: list[RecommendationResponse]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the recommender on application startup."""
    global recommender

    logger.info("Starting Movie Recommender API...")

    try:
        model_path = Config.MODELS_DIR / "svd_recommender.pkl"

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}")
            logger.warning("API will start but recommendations will not be available")
            logger.warning("Please train a model first using: python src/models/train_model.py")
            recommender = None
        else:
            recommender = MovieRecommender(model_path)
            logger.info("Recommender initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing recommender: {e}")
        recommender = None


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on application shutdown."""
    logger.info("Shutting down Movie Recommender API...")


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """
    Check if the API is running and the model is loaded.

    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy" if recommender is not None else "degraded",
        model_loaded=recommender is not None,
        version="0.1.0",
    )


# Recommendation endpoints
@app.get(
    "/recommend/{user_id}",
    response_model=UserRecommendationsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Recommendations"],
    summary="Get personalized movie recommendations",
)
async def get_recommendations(
    user_id: int = Path(..., description="User ID to get recommendations for", ge=1),
    k: int = Query(
        Config.DEFAULT_K_RECOMMENDATIONS,
        description="Number of recommendations to return",
        ge=1,
        le=100,
    ),
    exclude_rated: bool = Query(
        True,
        description="Whether to exclude movies the user has already rated",
    ),
) -> UserRecommendationsResponse:
    """
    Generate personalized movie recommendations for a specific user.

    This endpoint uses collaborative filtering to predict which movies the user would enjoy
    based on their past ratings and the ratings of similar users.

    **Parameters:**
    - **user_id**: The ID of the user to generate recommendations for
    - **k**: Number of recommendations to return (default: 10, max: 100)
    - **exclude_rated**: Whether to exclude movies already rated by the user (default: true)

    **Returns:**
    A list of recommended movies with predicted ratings and metadata.

    **Cold Start Handling:**
    If the user is new or the model cannot generate predictions, the API falls back
    to returning popular movies.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not available. Model may not be trained yet.",
        )

    try:
        recommendations = recommender.recommend_for_user(
            user_id=user_id,
            k=k,
            exclude_rated=exclude_rated,
        )

        return UserRecommendationsResponse(
            user_id=user_id,
            k=k,
            recommendations=[
                RecommendationResponse(**rec) for rec in recommendations
            ],
        )

    except ValueError as e:
        logger.error(f"Invalid user_id {user_id}: {e}")
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/similar/{movie_id}",
    response_model=SimilarMoviesResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Movie not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Recommendations"],
    summary="Get similar movies",
)
async def get_similar_movies(
    movie_id: int = Path(..., description="Movie ID to find similar movies for", ge=1),
    k: int = Query(
        10,
        description="Number of similar movies to return",
        ge=1,
        le=50,
    ),
) -> SimilarMoviesResponse:
    """
    Find movies similar to a given movie based on latent factors.

    This endpoint uses the learned latent representations from the collaborative filtering
    model to find movies that are similar in the latent space.

    **Parameters:**
    - **movie_id**: The ID of the reference movie
    - **k**: Number of similar movies to return (default: 10, max: 50)

    **Returns:**
    A list of similar movies with similarity scores.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not available. Model may not be trained yet.",
        )

    try:
        similar_movies = recommender.get_similar_movies(movie_id=movie_id, k=k)

        if not similar_movies:
            raise HTTPException(
                status_code=404,
                detail=f"Movie {movie_id} not found or no similar movies available",
            )

        return SimilarMoviesResponse(
            movie_id=movie_id,
            k=k,
            similar_movies=[
                RecommendationResponse(**movie) for movie in similar_movies
            ],
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error finding similar movies for {movie_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/predict/{user_id}/{movie_id}",
    tags=["Predictions"],
    summary="Predict rating for a user-movie pair",
)
async def predict_rating(
    user_id: int = Path(..., description="User ID", ge=1),
    movie_id: int = Path(..., description="Movie ID", ge=1),
) -> dict[str, Any]:
    """
    Predict the rating a user would give to a specific movie.

    **Parameters:**
    - **user_id**: The ID of the user
    - **movie_id**: The ID of the movie

    **Returns:**
    The predicted rating (0.5-5.0 scale).
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not available. Model may not be trained yet.",
        )

    try:
        predicted_rating = recommender.predict_rating(user_id, movie_id)

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "predicted_rating": round(predicted_rating, 2),
        }

    except Exception as e:
        logger.error(f"Error predicting rating for user {user_id}, movie {movie_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Root endpoint
@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """
    Root endpoint with API information.

    Returns:
        Welcome message and links to documentation
    """
    return {
        "message": "Welcome to the Movie Recommender API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"},
    )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting API server on {Config.API_HOST}:{Config.API_PORT}")

    uvicorn.run(
        "src.api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD,
        log_level=Config.LOG_LEVEL.lower(),
    )
