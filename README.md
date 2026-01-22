# 🎬 Movie Recommendation System

Sistema de recomendação de filmes profissional e pronto para produção, **100% compatível com Python 3.14**.

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.5+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚡ Quick Start (3 minutos)

```cmd
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar dados e treinar modelo
python -m src.data.loader
python -m src.features.build_features
python -m src.models.train_model

# 3. Iniciar API
uvicorn src.api.main:app --reload

# 4. Testar no navegador
# Abra: http://localhost:8000/docs
# Teste: GET /recommend/1?k=5
```

**Pronto!** Sistema funcionando com 610 usuários e 9.742 filmes do MovieLens! 🎉

---

## 📋 Visão Geral

Sistema completo de recomendação que usa **Matrix Factorization (SVD)** para gerar recomendações personalizadas de filmes. Implementado com scikit-learn puro - **sem necessidade de compilação C/C++**.

### 🎯 Características Principais

- ✅ **Python 3.14 compatível** - Funciona sem Visual Studio Build Tools
- ✅ **Matrix Factorization** usando TruncatedSVD do scikit-learn
- ✅ **REST API** com FastAPI e documentação automática (Swagger)
- ✅ **MLflow** para tracking de experimentos e versionamento de modelos
- ✅ **Cold start strategy** para novos usuários (filmes populares)
- ✅ **Filmes similares** baseados em fatores latentes (cosine similarity)
- ✅ **Type hints** em 100% do código
- ✅ **Logging estruturado** para debugging
- ✅ **Testes** com pytest

### 📊 Dataset MovieLens

O sistema utiliza o **MovieLens Small Dataset**, um conjunto de dados real de avaliações de filmes:

| Métrica | Valor |
|---------|-------|
| 📈 Avaliações | 100.836 |
| 👥 Usuários | 610 (IDs de 1 a 610) |
| 🎬 Filmes | 9.742 |
| 📅 Período | 1995 - 2018 |
| ⭐ Escala | 0.5 a 5.0 estrelas |

**⚠️ Importante**: Os "usuários" (IDs 1-610) são usuários reais do dataset MovieLens, não usuários que você cadastrou. O modelo já foi treinado com o histórico de avaliações deles.

---

## 🧠 Como Funciona

### Fluxo do Sistema

```
┌─────────────────────────────────────────────┐
│  1. DADOS                                   │
│  MovieLens Dataset (100k avaliações)        │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                     │
│  - Matriz user-item (610 × 9742)            │
│  - Cálculo de bias (média por usuário)      │
│  - Estatísticas (popularidade, etc)         │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  3. TREINAMENTO                             │
│  - TruncatedSVD (100 fatores latentes)      │
│  - Decomposição matricial                   │
│  - RMSE: 0.95 | MAE: 0.74                   │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  4. PREDIÇÃO                                │
│  - Recomendações personalizadas             │
│  - Filmes similares                         │
│  - Predição de notas                        │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  5. API REST                                │
│  FastAPI + Swagger Docs                     │
│  http://localhost:8000/docs                 │
└─────────────────────────────────────────────┘
```

### Exemplo Prático

```
Usuário 1 já avaliou 232 filmes no dataset
(Ex: Toy Story: 4.0★, Jumanji: 3.5★, Matrix: 5.0★)
    ↓
Modelo aprende que ele gosta de:
- Sci-Fi / Ação / Aventura
- Filmes dos anos 90-2000
    ↓
API recomenda filmes similares que ele NÃO viu:
1. Inception (2010) - Previsto: 4.8★
2. Interstellar (2014) - Previsto: 4.7★
3. The Prestige (2006) - Previsto: 4.6★
```

---

## 🎯 Uso da API

### Documentação Interativa

Após iniciar a API, acesse:

- **Swagger UI**: http://localhost:8000/docs (recomendado)
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Principais

#### 1️⃣ Obter Recomendações

```bash
GET /recommend/{user_id}?k=10&exclude_rated=true
```

**Exemplo Python:**
```python
import requests

response = requests.get("http://localhost:8000/recommend/1?k=5")
data = response.json()

for movie in data['recommendations']:
    print(f"🎬 {movie['title']} - ⭐{movie['predicted_rating']}")
```

**Resposta:**
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
    }
  ]
}
```

#### 2️⃣ Filmes Similares

```bash
GET /similar/{movie_id}?k=5
```

#### 3️⃣ Prever Nota

```bash
GET /predict/{user_id}/{movie_id}
```

#### 4️⃣ Health Check

```bash
GET /health
```

---

## 📊 Modelo de Machine Learning

### Algoritmo: Matrix Factorization (SVD)

**Fórmula:**
```
rating_previsto = user_mean + (user_factors · item_factors)
```

### Métricas de Performance

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **RMSE** | 0.95 | Erro médio de ~0.95 estrelas |
| **MAE** | 0.74 | Desvio médio absoluto |
| **Variância Explicada** | 66.73% | Modelo captura 2/3 dos padrões |
| **Sparsity** | 98.53% | Usuários avaliaram ~1.5% dos filmes |
| **Tempo de Treino** | ~20-30s | Em CPU comum |

---

## ❓ Perguntas Frequentes (FAQ)

**P: De onde vêm os usuários (1-610)?**  
R: Do dataset MovieLens! São usuários reais que avaliaram filmes entre 1995-2018.

**P: Posso adicionar meus próprios usuários?**  
R: Sim! Adicione avaliações ao dataset e re-treine o modelo.

**P: O que significa "Usuário 1"?**  
R: É o primeiro usuário do dataset MovieLens (ID=1), que já avaliou 232 filmes.

**P: Funciona sem internet?**  
R: Após o download inicial do dataset, sim! A API roda completamente offline.

**P: Precisa de GPU?**  
R: Não! Funciona perfeitamente em CPU comum.

**P: Como ver o histórico de um usuário?**
```python
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
user_history = df[df.userId == 1][['title', 'rating']]
print(user_history)
```

---

## 🛠️ Stack Tecnológica

| Tecnologia | Versão | Propósito |
|-----------|---------|-----------|
| Python | 3.14+ | Linguagem |
| scikit-learn | 1.5+ | Matrix Factorization |
| FastAPI | 0.115+ | REST API |
| Pandas | 2.2+ | Manipulação dados |
| NumPy | 2.0+ | Operações matriciais |
| MLflow | 2.16+ | Experiment tracking |

---

## 📁 Estrutura do Projeto

```
movie-recommender-system/
├── data/
│   ├── raw/              # MovieLens original
│   ├── processed/        # Dados processados
│   └── output/models/    # Modelos treinados
├── src/
│   ├── data/            # Carregamento de dados
│   ├── features/        # Feature engineering
│   ├── models/          # Treinamento e predição
│   └── api/             # FastAPI REST API
├── tests/               # Testes unitários
└── requirements.txt     # Dependências
```

---

## 🧪 Testes

```bash
# Rodar todos os testes
pytest

# Com coverage
pytest --cov=src --cov-report=html
```

---

## 📈 MLflow

```bash
mlflow ui
```

Acesse: http://localhost:5000

---

## 📚 Documentação Adicional

- 📘 [INSTALL.md](INSTALL.md) - Guia detalhado de instalação
- 📗 [API.md](API.md) - Documentação completa da API
- 📕 [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitetura técnica
- 📙 [QUICKSTART.md](QUICKSTART.md) - Início rápido

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📝 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes

---

## 🙏 Agradecimentos

- **GroupLens Research** - Dataset MovieLens
- **Scikit-learn** - Biblioteca de ML
- **FastAPI** - Framework web moderno
- **MLflow** - Plataforma de MLOps

---

## 👤 Desenvolvido por

**Kevin**

Projeto educacional de Sistema de Recomendação usando Matrix Factorization e Python 3.14.

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela! ⭐**

Made with ❤️ and Python 3.14

</div>
