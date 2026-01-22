@echo off
echo ========================================
echo   Movie Recommender - Python 3.14
echo   Instalacao Automatica
echo ========================================
echo.

echo [1/6] Criando ambiente virtual...
python -m venv venv
if errorlevel 1 goto error

echo.
echo [2/6] Ativando ambiente...
call venv\Scripts\activate.bat

echo.
echo [3/6] Atualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 goto error

echo.
echo [4/6] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [5/6] Verificando instalacao...
python -c "import pandas, numpy, sklearn, fastapi, mlflow; print('Todas as bibliotecas instaladas!')"
if errorlevel 1 goto error

echo.
echo [6/6] Baixando dados do MovieLens...
python -m src.data.loader
if errorlevel 1 (
    echo Aviso: Falha ao baixar dados. Voce pode baixar manualmente depois.
)

echo.
echo ========================================
echo   ✓ Instalacao Completa!
echo ========================================
echo.
echo Proximos passos:
echo.
echo 1. venv\Scripts\activate.bat
echo 2. python -m src.features.build_features
echo 3. python -m src.models.train_model
echo 4. uvicorn src.api.main:app --reload
echo.
echo Depois acesse: http://localhost:8000/docs
echo.
pause
goto end

:error
echo.
echo ========================================
echo   ❌ Erro na instalacao!
echo ========================================
echo.
echo Verifique:
echo - Python 3.10+ esta instalado?
echo - Tem conexao com internet?
echo - Esta no diretorio correto?
echo.
pause
goto end

:end
EOF