"""
Cenário 4 — Busca de hiperparâmetros (Keras Tuner RandomSearch), gerando a
geração "_tuned".

Contexto (NOTAS.md decisões 19-25)
-------------------------------------
Com o filtro de confiança (cenários 2 e 3) não produzindo uma vantagem
estatisticamente confiável, este cenário ataca a segunda hipótese razoável:
talvez a arquitetura/configuração de treino usada (fixada manualmente nas
decisões 1-11) não seja a melhor possível para este problema. Busca-se,
por (mercado, arquitetura), a combinação de hiperparâmetros que minimiza
`val_loss` (MSE) — a mesma métrica que `EarlyStopping` já monitora no
treino normal, mantendo a busca barata e diretamente comparável ao
baseline "_log".

Escopo da busca: parâmetros de arquitetura (filtros/unidades Conv1D/LSTM,
kernel_size, unidades Dense, dropout) e de treino (learning_rate do Adam,
batch_size, patience do EarlyStopping) — ver `src/models.py` para os
ranges exatos de cada um (deliberadamente amplos, não uma vizinhança
estreita dos valores originais). `TIMEFRAME` (janela de 60 barras) fica
FORA do escopo: mudar o formato de entrada invalidaria as Tabelas 1-3 do
artigo, que documentam especificamente a arquitetura `(60, 5)`.

Método: `keras_tuner.RandomSearch`, 18 trials por busca × 6 combinações
mercado/arquitetura = até ~108 treinos completos (cada trial ainda usa
EarlyStopping, então a maioria para bem antes de 200 épocas). Tempo de
execução observado: ~6 horas em CPU.

Fluxo
-----
1. Para cada uma das 6 combinações (mercado, arquitetura), roda
   `tune_one_model` — RandomSearch otimizando val_loss sobre o split
   treino/validação (features "_log", 5 colunas).
2. `apply_best_hp` retreina um modelo final com a configuração vencedora
   (não reaproveita os pesos do melhor trial, retreina do zero para manter
   o mesmo fluxo de treino/salvamento das outras gerações) e salva como
   geração "_tuned" — os modelos "_log" originais NÃO são sobrescritos,
   permanecem como baseline de comparação.

O que este script produz
-------------------------
- `models/{market}/{architecture}_tuned.keras` — os 6 modelos retreinados
  com os hiperparâmetros vencedores.
- `models/{market}/scaler_tuned.pkl` — 1 scaler por mercado.
- `models/{market}/{architecture}_best_hp.json` — os hiperparâmetros
  vencedores de cada busca, em formato JSON.
- `tuning/` — diretório de scratch do Keras Tuner (logs/checkpoints por
  trial); não é um artefato final, é gitignored.

Avaliação/backtest da geração "_tuned" NÃO estão incluídos neste script —
rode-os separadamente (mesma forma do cenário 1, trocando o suffix):

    python -c "from src.evaluate import evaluate_all; evaluate_all(suffix='_tuned')"
    python -c "from src.backtest import backtest_all; backtest_all(suffix='_tuned')"

Resultado obtido (referência — ver NOTAS.md decisão 25 para a análise
completa): acurácia direcional permaneceu ~0,49-0,50 (praticamente
inalterada frente a "_log"); Sharpe permaneceu negativo nas 12
combinações, 11/12 estatisticamente significativas (o achado de perda
consistente ficou MAIS robusto, não menos). Terceira intervenção
independente sem ganho, após os cenários 2 e 3.

Como executar
-------------
    python -m src.pipelines.04_hyperparameter_tuning

Aviso: esta é a etapa mais demorada do experimento (~6 horas em CPU para
as 6 buscas completas). Considere rodar em background ou com um
max_trials menor para um teste rápido, ex.:
    python -c "from src.tune import tune_all; tune_all(max_trials=3)"

Pré-requisitos: CSVs brutos em data/dados_{WDON,WINN}_H1.csv. Não depende
dos modelos da geracao "_log" (cada busca treina do zero).
"""

from src.tune import tune_all, MAX_TRIALS

SUFFIX = '_tuned'


def run():
  print("\n" + "=" * 70)
  print("CENARIO 4: Busca de hiperparametros (geracao '_tuned')")
  print(f"Trials por busca: {MAX_TRIALS} | Combinacoes: 6 (3 arquiteturas x 2 mercados)")
  print("Aviso: esta etapa pode levar horas em CPU.")
  print("=" * 70)

  results = tune_all(suffix=SUFFIX)

  print("\nConcluido. Hiperparametros vencedores:")
  for (market, architecture), hp in results.items():
    print(f"  {market}/{architecture}: {hp}")

  print("\nPara avaliar/testar em backtest a geracao '_tuned', rode:")
  print("  python -c \"from src.evaluate import evaluate_all; evaluate_all(suffix='_tuned')\"")
  print("  python -c \"from src.backtest import backtest_all; backtest_all(suffix='_tuned')\"")


if __name__ == "__main__":
  run()
