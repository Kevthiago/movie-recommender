# 📡 API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

**GET** `/health`

Verifica se a API está funcionando e o modelo está carregado.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### Get Recommendations

**GET** `/recommend/{user_id}`

Retorna recomendações personalizadas para um usuário.

**Parameters:**
- `user_id` (path, required): ID do usuário
- `k` (query, optional): Número de recomendações (default: 10, max: 100)
- `exclude_rated` (query, optional): Excluir filmes já avaliados (default: true)

**Example:**
```bash
GET /recommend/1?k=5&exclude_rated=true
```

**Response:**
```json
{
  "user_id": 1,
  "k": 5,
  "recommendations": [
    {
      "movie_id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "predicted_rating": 4.85
    },
    ...
  ]
}
```

**Status Codes:**
- `200`: Success
- `404`: User not found
- `503`: Model not available

---

### Get Similar Movies

**GET** `/similar/{movie_id}`

Encontra filmes similares baseado em fatores latentes.

**Parameters:**
- `movie_id` (path, required): ID do filme de referência
- `k` (query, optional): Número de filmes similares (default: 10, max: 50)

**Example:**
```bash
GET /similar/1?k=5
```

**Response:**
```json
{
  "movie_id": 1,
  "k": 5,
  "similar_movies": [
    {
      "movie_id": 2858,
      "title": "American Beauty (1999)",
      "genres": "Drama|Romance",
      "similarity_score": 0.923
    },
    ...
  ]
}
```

**Status Codes:**
- `200`: Success
- `404`: Movie not found

---

### Predict Rating

**GET** `/predict/{user_id}/{movie_id}`

Prediz qual nota um usuário daria para um filme específico.

**Parameters:**
- `user_id` (path, required): ID do usuário
- `movie_id` (path, required): ID do filme

**Example:**
```bash
GET /predict/1/100
```

**Response:**
```json
{
  "user_id": 1,
  "movie_id": 100,
  "predicted_rating": 4.12
}
```

---

## Error Responses

Todos os endpoints podem retornar erros no formato:

```json
{
  "detail": "Error message here"
}
```

**Common Status Codes:**
- `400`: Bad Request - Parâmetros inválidos
- `404`: Not Found - Recurso não encontrado
- `500`: Internal Server Error - Erro no servidor
- `503`: Service Unavailable - Modelo não carregado

---

## Interactive Documentation

Acesse a documentação interativa em:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Permite testar os endpoints diretamente no navegador!

---

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get recommendations
user_id = 1
response = requests.get(f"{BASE_URL}/recommend/{user_id}?k=5")
recommendations = response.json()

for rec in recommendations["recommendations"]:
    print(f"{rec['title']}: {rec['predicted_rating']:.2f}")

# Similar movies
movie_id = 1
response = requests.get(f"{BASE_URL}/similar/{movie_id}?k=5")
similar = response.json()

# Predict rating
response = requests.get(f"{BASE_URL}/predict/{user_id}/{movie_id}")
prediction = response.json()
print(f"Predicted rating: {prediction['predicted_rating']}")
```

---

## Rate Limiting

Atualmente não há rate limiting, mas recomenda-se:
- Máximo 100 requisições por minuto por IP
- Use cache para requisições repetidas

---

## CORS

A API permite requisições de qualquer origem (CORS enabled).

Para produção, configure origens específicas em `src/api/main.py`.
