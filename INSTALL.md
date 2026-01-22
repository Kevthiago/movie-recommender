# 🚀 Guia de Instalação

## Requisitos

- Python 3.10 ou superior (testado no 3.14)
- pip ou Poetry

## Instalação Rápida

### Windows

```cmd
# 1. Navegar para o projeto
cd movie-recommender-system

# 2. Criar ambiente virtual
python -m venv venv

# 3. Ativar ambiente
venv\Scripts\activate.bat

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Verificar instalação
python -c "import pandas, numpy, sklearn, fastapi; print('✓ OK!')"
```

### Linux/Mac

```bash
# 1. Navegar para o projeto  
cd movie-recommender-system

# 2. Criar ambiente virtual
python -m venv venv

# 3. Ativar ambiente
source venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Verificar
python -c "import pandas, numpy, sklearn, fastapi; print('✓ OK!')"
```

## Pipeline Completo

```bash
# 1. Baixar dataset MovieLens
python -m src.data.loader

# 2. Processar e criar features
python -m src.features.build_features

# 3. Treinar modelo
python -m src.models.train_model

# 4. Iniciar API
uvicorn src.api.main:app --reload
```

## Acesse a API

- Documentação: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Script Automático

### Windows
```cmd
INSTALL.bat
```

## Troubleshooting

### Erro: Module not found

Certifique-se de estar no ambiente virtual:
```bash
# Windows
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### Erro ao baixar dados

Verifique sua conexão com internet. O script baixa ~6MB do MovieLens.

### Porta 8000 em uso

Mude a porta:
```bash
uvicorn src.api.main:app --reload --port 8001
```

## VS Code

1. Abra a pasta do projeto
2. Pressione `Ctrl+Shift+P`
3. Digite: "Python: Select Interpreter"
4. Escolha: `.\venv\Scripts\python.exe`

## Próximos Passos

Consulte `README.md` para detalhes sobre uso da API e exemplos.
