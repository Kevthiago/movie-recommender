# 0001 — Python 3.12 e uv como gerenciador de ambiente

- **Data:** 2026-10-30
- **Status:** aceita

## Contexto

O projeto dizia ser "100% compatível com Python 3.14", mas a máquina tinha Python 3.11, nenhuma
dependência instalada, e o `pyproject.toml` (Poetry, `python = "^3.10"`) contradizia o README.
As dependências estavam duplicadas em `requirements.txt` e `pyproject.toml`, sem versões travadas.
Resultado: não dava para garantir que dois computadores rodassem o mesmo código com as mesmas libs.

## Decisão

1. **Python 3.12**, fixado em `.python-version` e em `requires-python = ">=3.12,<3.13"`.
2. **uv** gerencia Python, ambiente virtual e dependências.
3. **`pyproject.toml` (padrão PEP 621) é a única fonte de dependências**; `uv.lock` trava as versões.
   `requirements.txt` e `INSTALL.bat` foram removidos.
4. Dependências de apoio separadas em grupos (`notebooks`, `test`, `dev`).
5. Removidas dependências não usadas pelo código (`sqlalchemy`, `pytest-asyncio`, `notebook` — já incluso
   em `jupyter`). Adicionada `pyarrow`, que o pandas exige para ler/escrever Parquet e antes só vinha por acaso
   como dependência transitiva do MLflow.

## Alternativas consideradas

| Opção | Por que não |
|---|---|
| pip + `requirements.txt` | Sem lockfile nativo (só com `pip-tools`), não gerencia a versão do Python, lento. Bom para aprender o básico, mas não é o padrão atual de mercado em projetos novos. |
| Poetry | Funciona bem, mas é mais lento, usa um formato próprio (`[tool.poetry]`) em vez do padrão PEP 621 e não instala o Python. |
| conda | Útil quando há dependências nativas pesadas (CUDA, GDAL). Não é o nosso caso. |
| Python 3.13/3.14 | Mais novo, mas parte do ecossistema científico chega com atraso a versões recentes. 3.12 é a versão "segura" com suporte longo. |

## Consequências

- `uv sync` reproduz exatamente o mesmo ambiente em qualquer máquina.
- Todo comando passa a ser `uv run <comando>` (ou ativar `.venv` manualmente).
- `INSTALL.md`, `QUICKSTART.md` e `INDEX.md` ainda citam pip; serão consolidados em `docs/` na Fase 1.
- `[tool.uv] package = false` por enquanto: o código ainda é o pacote `src`. Na Fase 1 ele vira
  `src/movie_recommender/`, um pacote instalável.
