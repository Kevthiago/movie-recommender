# CLAUDE.md

Regras de trabalho para o Claude neste repositório. Leia antes de qualquer tarefa.

## Propósito do projeto

Sistema de recomendação de filmes (MovieLens + Matrix Factorization + FastAPI + MLflow).
**O objetivo principal é aprendizado.** O Kevin quer entender cada decisão, principalmente a parte
de inteligência artificial (dados, features, modelagem, treino, avaliação, predição).
O projeto **não** deve virar "vibe code": nada entra sem que o raciocínio esteja explicado.

## Como trabalhamos

1. **Por fases.** Cada fase tem um objetivo claro, é discutida antes e revisada depois.
   Não avance para a próxima fase sem o Kevin aprovar.
2. **Mudanças pequenas e revisáveis.** Prefira vários commits pequenos e temáticos a um commit grande.
3. **Refatoração ≠ mudança de comportamento.** Ao reorganizar código, o comportamento deve continuar
   idêntico. Mudanças de comportamento (incluindo correções de bugs) são etapas separadas e explícitas.
4. **Testes antes de mexer.** Antes de refatorar algo sem testes, escreva testes de caracterização
   que fixem o comportamento atual.

## Regra de estudo (obrigatória)

Toda decisão técnica relevante de ML/dados deve vir acompanhada de um bloco **📚 Ponto de estudo**
na resposta, contendo:

- **O que** foi decidido (ex.: "split temporal em vez de aleatório").
- **Por que** — a intuição e, quando fizer sentido, a matemática (fórmulas em notação simples).
- **Alternativas** consideradas e por que foram descartadas.
- **Como verificar** — que número, gráfico ou teste mostra que a decisão foi boa.
- **Para se aprofundar** — 1 a 3 referências (paper, documentação, capítulo de livro).

Decisões que valem um registro permanente vão para `docs/` (ver "Documentação de estudo" abaixo).

Prefira implementar algoritmos centrais de forma didática (NumPy explícito, com comentários
explicando cada passo matemático) antes de trocar por uma biblioteca pronta. Quando usar uma
biblioteca, explique o que ela faz por baixo.

## Quando parar e perguntar

**Execução é autônoma**: rodar comandos, instalar dependências, rodar testes, ler/editar arquivos
não exigem confirmação.

**Decisões são do Kevin.** Pare e pergunte (com opções e uma recomendação) quando a escolha for:

- de modelagem (algoritmo, função de perda, hiperparâmetros de partida, estratégia de split, métricas);
- de arquitetura ou organização (estrutura de pastas, nomes de pacotes, contratos da API);
- de ferramental (gerenciador de pacotes, versão do Python, novas dependências relevantes);
- um trade-off que muda o resultado ou o que ele vai estudar.

Também confirme antes de ações difíceis de desfazer: `git push`, reescrita de histórico,
apagar dados/modelos treinados.

## Documentação de estudo

- `docs/decisions/NNNN-titulo.md` — registros de decisão (ADR): contexto, decisão, alternativas, consequências.
- `docs/study/NN-tema.md` — material de estudo produzido junto com o desenvolvimento
  (ex.: `01-matrix-factorization.md`, `02-metricas-de-avaliacao.md`).

Escreva a documentação em português. Código, nomes de variáveis e docstrings em inglês.

## Ambiente

- Python **3.12** (fixado em `.python-version`), gerenciado pelo **uv**. `pyproject.toml` é a única
  fonte de dependências; `uv.lock` trava as versões exatas e deve ser commitado.
- Adicionar dependência: `uv add <pacote>` (ou `uv add --group <grupo> <pacote>`). Nunca editar `uv.lock` à mão.
- Grupos: `notebooks` (jupyter, gráficos), `test` (pytest), `dev` (ambos, instalado por padrão).
- No Windows o uv fica em `C:\Users\Kevin\.local\bin`; se o comando não for encontrado, adicione ao PATH.
- `MLFLOW_DISABLE_AGENT_HINT=1` silencia um aviso do MLflow 3 no início de cada execução.

## Comandos (estado atual, antes da refatoração da Fase 1)

```bash
uv sync                                       # cria .venv e instala tudo
uv run python -m src.data.loader              # baixa MovieLens -> data/raw, salva parquet em data/processed
uv run python -m src.features.build_features  # gera data/processed/features.parquet
uv run python -m src.models.train_model       # treina, avalia, salva data/output/models/svd_recommender.pkl
uv run uvicorn src.api.main:app --reload      # API em http://localhost:8000/docs
uv run pytest                                 # testes
uv run mlflow ui --backend-store-uri sqlite:///data/output/mlflow.db   # experimentos em http://localhost:5000
```

Atualize esta seção sempre que os comandos mudarem.

## Estado atual (atualizar a cada fase)

- **Fase 0 (setup) concluída em 2026-10-30.** Pipeline roda de ponta a ponta.
- Linha de base medida (split aleatório 80/20, `random_state=42`):
  TruncatedSVD RMSE **0.9478** · média do usuário 0.9497 · `μ + b_u + b_i` 0.8821.
  Ver `docs/study/00-ambiente-e-linha-de-base.md`.
- Testes: 5 passam, 2 skip, 3 quebram pelo bug `"small"` vs `"latest-small"` (será corrigido na Fase 1).
