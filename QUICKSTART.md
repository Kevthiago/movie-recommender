# ⚡ Quick Start Guide

## 🎯 Objetivo

Ter o sistema funcionando em **5 minutos**.

## 📋 Checklist Pré-requisitos

- [ ] Python 3.14 instalado
- [ ] pip instalado
- [ ] Conexão com internet

## 🚀 Passos Rápidos

### Windows

```cmd
# 1. Clonar o repositório
# Salvar o projeto na sua máquina

# 2. Abrir terminal na pasta
cd movie-recommender-clean

# 3. Rodar instalador automático
INSTALL.bat

# 4. Aguardar instalação completar
# O script faz tudo automaticamente!

# 5. Quando terminar, rodar:
venv\Scripts\activate.bat
python -m src.features.build_features
python -m src.models.train_model
uvicorn src.api.main:app --reload

# 6. Abrir navegador
# http://localhost:8000/docs
```

### Linux/Mac

```bash
# 1. Clonar
Salvar o projeto na sua máquina
cd movie-recommender-clean

# 2. Instalar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Pipeline
python -m src.data.loader
python -m src.features.build_features
python -m src.models.train_model

# 4. API
uvicorn src.api.main:app --reload

# 5. Abrir
# http://localhost:8000/docs
```

## ✅ Verificação

Se tudo funcionou, você verá:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Abra http://localhost:8000/docs e teste!

## 🧪 Testar API

### No navegador

1. Abra http://localhost:8000/docs
2. Clique em `GET /recommend/{user_id}`
3. Clique em "Try it out"
4. Digite `1` no campo user_id
5. Clique em "Execute"
6. Veja as recomendações!

### Via curl

```bash
curl http://localhost:8000/recommend/1?k=5
```

### Via Python

```python
import requests

r = requests.get("http://localhost:8000/recommend/1?k=5")
print(r.json())
```

## 📂 Arquivos Importantes

```
movie-recommender-clean/
├── INSTALL.bat          ← Windows: rode este!
├── requirements.txt     ← Dependências Python
├── README.md           ← Documentação completa
├── API.md              ← Documentação da API
└── src/
    ├── data/loader.py      ← Baixa dados
    ├── features/           ← Processa features
    ├── models/train_model.py ← Treina modelo
    └── api/main.py         ← API REST
```

## 🐛 Problemas Comuns

### "python: command not found"
→ Python não instalado ou não está no PATH

### "Module not found"
→ Ative o ambiente virtual primeiro:
```cmd
venv\Scripts\activate.bat  # Windows
source venv/bin/activate   # Linux/Mac
```

### "Port 8000 already in use"
→ Use outra porta:
```cmd
uvicorn src.api.main:app --reload --port 8001
```

### Erro ao baixar dados
→ Verifique conexão com internet

## 📞 Próximos Passos

1. ✅ Leia `README.md` para entender o sistema
2. ✅ Leia `API.md` para ver todos os endpoints
3. ✅ Explore `ARCHITECTURE.md` para detalhes técnicos
4. ✅ Modifique `src/config.py` para customizar

## 🎉 Pronto!

Você tem um sistema de recomendação funcionando!

Teste diferentes usuários:
- http://localhost:8000/recommend/1
- http://localhost:8000/recommend/50
- http://localhost:8000/recommend/100

Encontre filmes similares:
- http://localhost:8000/similar/1
- http://localhost:8000/similar/50
