# 🏗️ Architecture Documentation

## System Overview

O Movie Recommender System é construído em 3 camadas principais:

```
┌─────────────────────────────────────┐
│     APPLICATION LAYER (API)         │
│  FastAPI + Uvicorn + Auto Docs      │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     ML CORE (Training & Inference)  │
│  Matrix Factorization + MLflow      │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     DATA LAYER (ETL Pipeline)       │
│  MovieLens → Pandas → Parquet       │
└─────────────────────────────────────┘
```

## Tech Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.14+ | Core language |
| Data | Pandas | 2.2+ | Data manipulation |
| ML | Scikit-learn | 1.5+ | Matrix Factorization |
| API | FastAPI | 0.115+ | REST endpoints |
| Server | Uvicorn | 0.32+ | ASGI server |
| MLOps | MLflow | 2.16+ | Experiment tracking |

### Why These Choices?

- **Python 3.14**: Cutting edge, fully compatible
- **Scikit-learn**: No compilation needed, pure Python
- **FastAPI**: Modern, fast, auto-documentation
- **Pandas/NumPy 2.x**: Latest versions with better performance

## Data Flow

```
MovieLens API
     ↓
Download ZIP (6MB)
     ↓
Extract CSV files
     ↓
Pandas DataFrame
     ↓
Feature Engineering
     ↓
Parquet Files (optimized)
     ↓
Matrix Factorization
     ↓
Trained Model (.pkl)
     ↓
FastAPI Endpoints
     ↓
User Gets Recommendations
```

## Module Structure

### src/data/

**Purpose**: Data ingestion and loading

**Files**:
- `loader.py`: Downloads MovieLens, extracts, loads to Pandas

**Key Functions**:
```python
def download_dataset() -> Path
def load_ratings() -> pd.DataFrame
def load_movies() -> pd.DataFrame
```

### src/features/

**Purpose**: Feature engineering and transformation

**Files**:
- `build_features.py`: Creates features from raw data

**Features Created**:
- User statistics (avg rating, count, std)
- Movie statistics (avg rating, popularity)
- Temporal features (year, month, day)
- Genre features
- Popularity scores

### src/models/

**Purpose**: ML model training and inference

**Files**:
- `train_model.py`: Matrix Factorization training with MLflow
- `predict.py`: Generate recommendations and predictions

**Key Classes**:
```python
class MatrixFactorizationRecommender:
    def fit(ratings_df) -> self
    def predict(user_id, movie_id) -> float
    def recommend(user_id, k) -> List[dict]
```

### src/api/

**Purpose**: REST API endpoints

**Files**:
- `main.py`: FastAPI application with all endpoints

**Endpoints**:
- `GET /health`: Health check
- `GET /recommend/{user_id}`: Get recommendations
- `GET /similar/{movie_id}`: Find similar movies
- `GET /predict/{user_id}/{movie_id}`: Predict rating

## Algorithm Details

### Matrix Factorization (TruncatedSVD)

**Mathematical Foundation**:

User-Item matrix **R** ≈ **U** × **Σ** × **V**ᵀ

Where:
- **U**: User factors (users × k)
- **Σ**: Singular values
- **V**ᵀ: Item factors (k × items)
- k: Number of latent factors (100 by default)

**Prediction Formula**:

```
r̂ᵤᵢ = μᵤ + qᵢᵀ × pᵤ
```

Where:
- r̂ᵤᵢ: Predicted rating for user u and item i
- μᵤ: User u's average rating (bias)
- qᵢ: Item i's latent factors
- pᵤ: User u's latent factors

**Training Process**:

1. Create user-item matrix (sparse)
2. Calculate user means (bias correction)
3. Center matrix by subtracting user means
4. Apply TruncatedSVD
5. Extract user and item factors
6. Save model

**Inference**:

1. Load model (user/item factors)
2. For prediction: dot product + user bias
3. For recommendations: compute all predictions, sort, return top-K
4. For similar items: cosine similarity of item factors

### Cold Start Strategy

**New User** (not in training set):
- Return most popular movies
- Based on weighted rating formula

**New Movie** (not in training set):
- Cannot make predictions
- Skip in recommendations

## Scalability

### Current Limitations

- **In-memory**: All data loaded in RAM
- **Single instance**: No horizontal scaling
- **Batch mode**: Re-train required for new data

### Scale-up Options

| Component | Current | Scalable Alternative |
|-----------|---------|---------------------|
| Data | Pandas | Dask, Spark |
| Model | In-memory | Redis cache |
| API | Single instance | Load balancer + replicas |
| Database | SQLite | PostgreSQL, MongoDB |
| Training | Single-core | Distributed (Ray) |

### Performance Metrics

**MovieLens Small** (100k ratings):
- Load data: ~2s
- Feature engineering: ~5s
- Training: ~20s
- Inference (single): ~1ms
- API latency (p50): ~50ms

## MLOps Pipeline

```
Data → Features → Train → Log (MLflow) → Save Model → Serve (API)
                     ↓
               Track metrics
               Log parameters
               Save artifacts
```

### MLflow Integration

**Tracked**:
- Parameters (n_factors, algorithm, etc)
- Metrics (RMSE, MAE, variance explained)
- Artifacts (model.pkl)
- Metadata (n_users, n_items)

**View experiments**:
```bash
mlflow ui
```

## Security Considerations

### Current State
- ✅ Input validation (Pydantic)
- ✅ CORS enabled (configurable)
- ❌ No authentication
- ❌ No rate limiting
- ❌ No API keys

### Production Recommendations
1. Add API key authentication
2. Implement rate limiting (e.g., slowapi)
3. Use HTTPS only
4. Restrict CORS origins
5. Add request logging
6. Implement caching (Redis)

## Monitoring

### Health Metrics to Monitor

**System**:
- CPU usage
- Memory usage
- Disk I/O

**Application**:
- Request count
- Response time (p50, p95, p99)
- Error rate
- Cache hit rate

**Model**:
- Prediction distribution
- RMSE drift
- Coverage (% users/items seen)
- Cold start rate

### Recommended Tools

- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger
- **Alerting**: PagerDuty

## Deployment Architecture

### Development
```
Local Machine → Python venv → FastAPI dev server
```

### Production (Example)
```
GitHub → CI/CD (GitHub Actions) → 
Docker Build → Container Registry → 
Kubernetes → Load Balancer → 
Users
```

## Future Enhancements

### Short-term
- [ ] User-based collaborative filtering (KNN)
- [ ] Content-based filtering (genres, tags)
- [ ] Hybrid recommender
- [ ] A/B testing framework

### Medium-term
- [ ] Real-time model updates
- [ ] Deep learning models (Neural CF)
- [ ] Context-aware recommendations
- [ ] Explainability (why this recommendation?)

### Long-term
- [ ] Multi-objective optimization
- [ ] Reinforcement learning
- [ ] Multi-modal recommendations
- [ ] Federated learning

## References

- **Paper**: "Matrix Factorization Techniques for Recommender Systems" (Koren et al.)
- **Dataset**: [MovieLens](https://grouplens.org/datasets/movielens/)
- **Framework**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Library**: [Scikit-learn](https://scikit-learn.org/)
