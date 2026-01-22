# 📚 Índice de Documentação

Bem-vindo ao Movie Recommender System! Este arquivo lista toda a documentação disponível.

## 🚀 Começando

1. **[QUICKSTART.md](QUICKSTART.md)** ⚡
   - Instalação e uso em 5 minutos
   - Para quem quer começar AGORA
   - Comandos básicos

2. **[INSTALL.md](INSTALL.md)** 🔧
   - Guia detalhado de instalação
   - Troubleshooting
   - Diferentes ambientes (Windows/Linux/Mac)

3. **[README.md](README.md)** 📖
   - Visão geral do projeto
   - Features principais
   - Exemplos de uso

## 🏗️ Arquitetura e Design

4. **[ARCHITECTURE.md](ARCHITECTURE.md)** 🏛️
   - Arquitetura do sistema (3 camadas)
   - Algoritmo de Matrix Factorization
   - Tech stack e decisões de design
   - Escalabilidade e deployment

## 🌐 API

5. **[API.md](API.md)** 📡
   - Documentação completa de todos os endpoints
   - Exemplos de requisições
   - Códigos de status
   - Cliente Python

## 📁 Estrutura do Projeto

```
movie-recommender-clean/
│
├── 📚 Documentação
│   ├── README.md           # Visão geral
│   ├── QUICKSTART.md       # Início rápido
│   ├── INSTALL.md          # Instalação
│   ├── API.md              # Documentação da API
│   ├── ARCHITECTURE.md     # Arquitetura técnica
│   └── INDEX.md            # Este arquivo
│
├── ⚙️ Configuração
│   ├── requirements.txt    # Dependências pip
│   ├── pyproject.toml      # Configuração Poetry
│   ├── .env.example        # Variáveis de ambiente
│   ├── .gitignore          # Git ignore rules
│   └── LICENSE             # Licença MIT
│
├── 🚀 Scripts
│   └── INSTALL.bat         # Instalador Windows
│
├── 📊 Data (Git-ignored)
│   ├── raw/                # Dados brutos MovieLens
│   ├── processed/          # Dados processados (Parquet)
│   └── output/             # Modelos treinados
│
├── 📓 Notebooks
│   └── 01_eda.ipynb        # Análise exploratória
│
├── 💻 Source Code
│   └── src/
│       ├── config.py           # Configuração centralizada
│       ├── logger.py           # Logging estruturado
│       ├── data/
│       │   └── loader.py       # Carregamento de dados
│       ├── features/
│       │   └── build_features.py # Feature engineering
│       ├── models/
│       │   ├── train_model.py  # Treinamento
│       │   └── predict.py      # Predição/Recomendação
│       └── api/
│           └── main.py         # FastAPI REST API
│
└── 🧪 Tests
    └── tests/
        └── test_data.py        # Testes unitários
```

## 📝 Guias por Tarefa

### Quero começar rapidamente
→ Leia [QUICKSTART.md](QUICKSTART.md)

### Quero entender como funciona
→ Leia [README.md](README.md) depois [ARCHITECTURE.md](ARCHITECTURE.md)

### Quero usar a API
→ Leia [API.md](API.md)

### Tenho problemas na instalação
→ Leia [INSTALL.md](INSTALL.md) seção "Troubleshooting"

### Quero contribuir
→ Leia [README.md](README.md) seção "Contribuindo"

### Quero fazer deploy
→ Leia [ARCHITECTURE.md](ARCHITECTURE.md) seção "Deployment"

## 🎓 Ordem Recomendada de Leitura

### Para Usuários
1. QUICKSTART.md - Começar
2. README.md - Entender o que é
3. API.md - Usar a API

### Para Desenvolvedores
1. README.md - Contexto geral
2. INSTALL.md - Setup do ambiente
3. ARCHITECTURE.md - Arquitetura técnica
4. Código em `src/` - Implementação

### Para Arquitetos/Tech Leads
1. ARCHITECTURE.md - Design e decisões
2. README.md - Features e stack
3. Código em `src/` - Padrões usados

## 🔗 Links Úteis

- **Dataset**: [MovieLens](https://grouplens.org/datasets/movielens/)
- **FastAPI Docs**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org/)
- **MLflow**: [mlflow.org](https://mlflow.org/)

## 📞 Suporte

Problemas? Verifique nesta ordem:

1. QUICKSTART.md → Passos básicos corretos?
2. INSTALL.md → Troubleshooting
3. GitHub Issues → Problema já reportado?
4. Criar nova issue → Descreva o problema

## 🎯 TL;DR

```bash
# Instalar
INSTALL.bat  # Windows
# ou
pip install -r requirements.txt  # Manual

# Rodar
python -m src.data.loader
python -m src.features.build_features
python -m src.models.train_model
uvicorn src.api.main:app --reload

# Testar
curl http://localhost:8000/recommend/1
```

Documentação completa? ✅  
Sistema funcionando? ✅  
Pronto para usar! 🚀
