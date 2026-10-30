# 00 — Ambiente reproduzível e linha de base

Este é o primeiro material de estudo do projeto. Ele cobre duas coisas da Fase 0:

1. por que tratamos o **ambiente** como parte do experimento;
2. o que os primeiros números do modelo **realmente** dizem.

---

## Parte 1 — Ambiente reproduzível

### O problema

Em ML, um resultado só vale se puder ser reproduzido. "Meu modelo deu RMSE 0.95" não quer dizer nada
se outra pessoa (ou você daqui a seis meses) roda o mesmo código e obtém outro número porque o
scikit-learn mudou de versão e o algoritmo randomizado passou a se comportar diferente.

Um resultado depende de quatro coisas:

| Ingrediente | Como fixamos |
|---|---|
| Código | git |
| Dados | dataset público versionado (`ml-latest-small`) — mais tarde podemos guardar um hash |
| Aleatoriedade | `random_state=42` no split e no SVD |
| **Ambiente** (Python + bibliotecas) | `.python-version` + `uv.lock` |

### Conceitos

- **Ambiente virtual (`.venv/`)**: uma pasta com um Python e as bibliotecas só deste projeto.
  Evita que projetos diferentes briguem por versões.
- **Dependência direta vs. transitiva**: nós pedimos `mlflow`; o `mlflow` puxa outras ~100 bibliotecas.
  Pedimos ~12 pacotes e o ambiente final tem centenas.
- **Restrição vs. versão exata**: `pandas>=2.2.0` no `pyproject.toml` é uma *restrição* (o que aceitamos).
  O `uv.lock` grava a *versão exata* que foi resolvida (hoje, `pandas 3.1.0`) de **todas** as
  dependências, inclusive transitivas. É o lockfile que garante reprodutibilidade.

### Na prática

```bash
uv sync             # cria/atualiza .venv exatamente conforme uv.lock
uv run pytest       # roda um comando dentro do .venv, sem precisar "ativar"
uv add scipy        # adiciona dependência: atualiza pyproject.toml E uv.lock
uv lock --upgrade   # atualiza versões travadas, de forma deliberada
```

> Pegadinha que encontramos: o código usa `df.to_parquet(...)`, que precisa da biblioteca `pyarrow`,
> mas ela não estava declarada. Funcionava "por sorte", porque o MLflow a instalava. Se o MLflow
> deixasse de depender dela, o projeto quebraria sem ninguém ter mexido no código.
> **Regra:** declare tudo o que você importa diretamente.

---

## Parte 2 — Linha de base: o modelo é bom?

### O número do README

Rodando o pipeline original, reproduzimos o que o README dizia:

```
TruncatedSVD (100 fatores)   RMSE 0.9478   MAE 0.7350
```

Mas **um número sozinho não diz nada**. RMSE 0.95 é bom ou ruim? Para responder, comparamos com
modelos ingênuos — as *baselines*. Usamos exatamente o mesmo split (80/20, `random_state=42`).

### As baselines

Notação: $r_{ui}$ é a nota do usuário $u$ para o filme $i$; $\hat r_{ui}$ é a previsão.

**1. Média global** — chuta a mesma nota para tudo:

$$\hat r_{ui} = \mu$$

**2. Média do usuário** — cada pessoa tem seu "nível" de exigência:

$$\hat r_{ui} = \bar r_u$$

**3. Vieses de usuário e de filme** (*bias baseline*, Koren 2009):

$$\hat r_{ui} = \mu + b_u + b_i$$

- $b_i$: quanto o filme $i$ costuma ficar acima/abaixo da média (*Um Sonho de Liberdade* tem $b_i > 0$).
- $b_u$: quanto o usuário $u$ costuma dar notas acima/abaixo da média (usuário "bonzinho" tem $b_u > 0$).

Calculamos com **regularização** ($\lambda = 10$):

$$b_i = \frac{\sum_{u}(r_{ui} - \mu)}{n_i + \lambda}$$

O $\lambda$ no denominador "puxa para zero" o viés de filmes com poucas avaliações. Um filme com uma
única nota 5.0 não deve ganhar um viés enorme: é pouca evidência. Com $n_i = 1$ e $\lambda = 10$, o viés
fica em cerca de 1/11 do que seria sem regularização. Com $n_i = 300$, quase não muda.

### Resultado

| Modelo | RMSE |
|---|---|
| Média global | 1.0488 |
| Média do usuário | 0.9497 |
| **TruncatedSVD (modelo atual, 100 fatores)** | **0.9478** |
| $\mu + b_u + b_i$ | **0.8821** |

### O que isso significa

1. **Os 100 fatores latentes do SVD quase não acrescentam nada** à média do usuário
   (0.9497 → 0.9478). O modelo atual é, na prática, "preveja a média do usuário".
2. **Um modelo de três termos, sem nada de "IA", ganha do SVD com folga.**

Por quê? No modelo atual, a matriz usuário×filme é 98,5% vazia, e os vazios viram 0, o que depois da
centralização equivale a dizer "o usuário daria a nota média dele". O SVD então gasta quase toda a sua
capacidade reconstruindo esses milhões de "zeros falsos", e o produto
$p_u \cdot q_i$ fica minúsculo: medimos $|p_u \cdot q_i|$ médio de **0,02 estrela** (percentil 99: 0,23).
Sobra só a média do usuário. Além disso, o modelo atual não tem $b_i$:
ele ignora que alguns filmes são simplesmente melhores que outros.

### Lições

- **Sempre comece por uma baseline.** É ela que dá significado à métrica do modelo complexo.
- **Valores ausentes não são zero.** "Não avaliou" ≠ "odiou" ≠ "achou mediano".
  Modelos de Matrix Factorization de verdade (Funk SVD, ALS) otimizam o erro **apenas nas notas observadas**.
  Vamos implementar isso na Fase 3.
- Essas comparações vão virar código de verdade (`models/baselines.py` + `evaluation/`) na Fase 2.
  Por enquanto, o script de medição foi descartável.

### Para se aprofundar

- Koren, Bell & Volinsky (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
  Seções "A basic matrix factorization model" e "Adding biases".
- Simon Funk (2006). *Netflix Update: Try This at Home* — o post que popularizou o "Funk SVD".
- Documentação do uv: <https://docs.astral.sh/uv/concepts/projects/>
