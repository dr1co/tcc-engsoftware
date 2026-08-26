"""
Cenário 1 — Baseline (geração "_log"): 5 features em log-retorno, sem tuning,
sem filtro de confiança.

Contexto (NOTAS.md decisões 1-16)
----------------------------------
Esta é a primeira geração funcional do pipeline de treino/teste, após a
correção do problema de extrapolação do MinMaxScaler (decisão 12): as 4
colunas de preço (open/high/low/close) são transformadas em log-retorno
`log(X_t / X_{t-1})` antes do escalonamento, em vez de usar o preço em
nível absoluto. `real_volume` permanece em nível bruto. Total: 5 features
de entrada, tensor `(60, 5)` — o mesmo formato documentado nas Tabelas 1-3
do artigo.

Todas as etapas seguintes do experimento (filtro de confiança, tuning,
indicadores técnicos) usam esta geração como ponto de comparação — é o
"placar a bater" de todo o restante do trabalho.

Fluxo
-----
1. Treino: para cada uma das 6 combinações (mercado × arquitetura), roda
   `train_one_model` — sanitização → log-retorno → split cronológico
   70/15/15 → MinMaxScaler (fit no treino) → janelamento (60 barras) →
   treino com EarlyStopping (patience=12, máx. 200 épocas).
2. Avaliação: `evaluate_all` recarrega os 6 modelos salvos, reconstrói o
   conjunto de teste, prevê o log-retorno, reconstrói o preço e calcula
   RMSE/MAE/acurácia direcional em unidade real de preço.
3. Backtest: `backtest_all` simula as 2 estratégias (single_bar, multi_bar)
   para os 6 modelos (12 combinações), sem filtro de confiança
   (`min_confidence_pct=100`), calculando Sharpe Ratio anualizado, Maximum
   Drawdown, VaR histórico 95% e um teste-t de significância sobre os
   retornos por trade.

O que este script produz
-------------------------
- `models/{WDON,WINN}/{cnn,lstm,hybrid}_log.keras` — os 6 modelos treinados.
- `models/{WDON,WINN}/scaler_log.pkl` — 1 scaler por mercado (compartilhado
  pelas 3 arquiteturas daquele mercado).
- `models/{WDON,WINN}/{cnn,lstm,hybrid}_log_metrics.json` — RMSE/MAE/acurácia
  direcional por modelo.
- `results/comparison_log.csv` — tabela comparativa das 6 combinações
  (métricas de erro).
- `results/backtest_comparison_log.csv` — tabela comparativa das 12
  combinações mercado×arquitetura×estratégia (métricas de negócio).
- `results/equity_curves/{market}_{architecture}_{strategy}_log.csv` — série
  de P&L por trade e acumulado, uma por combinação (12 arquivos).

Resultado obtido (referência, não normativo — ver NOTAS.md para a análise
completa): acurácia direcional ~0,49-0,50 em todos os 6 modelos; Sharpe
Ratio negativo nas 12 combinações, 10/12 estatisticamente significativas
(p < 0,05) — ou seja, o achado confiável é perda consistente, não uma
vantagem direcional real. Este resultado motivou as etapas seguintes
(cenários 2-6).

Como executar
-------------
    python -m src.pipelines.01_baseline_log

Pré-requisitos: CSVs brutos em data/dados_{WDON,WINN}_H1.csv.
"""

from src.train import train_all
from src.evaluate import evaluate_all
from src.backtest import backtest_all

SUFFIX = '_log'
FEATURE_SET = 'base'


def run():
  print("\n" + "=" * 70)
  print("CENARIO 1: Baseline (geracao '_log') - 5 features em log-retorno")
  print("=" * 70)

  print("\n--- Etapa 1/3: Treinamento dos 6 modelos ---")
  train_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\n--- Etapa 2/3: Avaliacao (RMSE, MAE, acuracia direcional) ---")
  evaluate_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\n--- Etapa 3/3: Backtest (Sharpe, Maximum Drawdown, VaR) ---")
  backtest_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\nConcluido. Ver results/comparison_log.csv e results/backtest_comparison_log.csv.")


if __name__ == "__main__":
  run()
