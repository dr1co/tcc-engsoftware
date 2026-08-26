"""
Cenário 5 — Engenharia de atributos com indicadores técnicos (RSI, SMA,
MACD), gerando a geração "_features".

Contexto (NOTAS.md decisões 26-27)
-------------------------------------
Com filtro de confiança (cenários 2-3) e tuning de hiperparâmetros
(cenário 4) ambos sem produzir uma vantagem estatisticamente confiável, os
dois "gargalos" mais óbvios (regra de decisão e capacidade do modelo) já
foram descartados. Este cenário ataca a terceira hipótese: talvez as 5
features de entrada atuais (OHLC em log-retorno + volume) simplesmente não
carreguem informação direcional suficiente, e adicionar indicadores
técnicos — já citados no Referencial Teórico do artigo [Mohammed 2022] —
ajude o modelo a capturar padrões que o preço bruto não expõe diretamente.

Indicadores adicionados (`src/features.py::add_technical_indicators`, via
`pandas_ta`), com períodos encurtados para se adequar ao horizonte
intradiário (candles H1, janela de 60 barras) em vez dos períodos-padrão
de gráfico diário:
  - RSI (7 períodos) — mantido em nível bruto (já é um oscilador limitado
    a [0, 100], sem risco de extrapolação de escala).
  - SMA (10 períodos) — convertida para log-retorno da própria SMA, pelo
    mesmo motivo de estacionariedade da decisão 12 (evitar extrapolação do
    MinMaxScaler).
  - MACD (rápida 6 / lenta 13 / sinal 4) — as 3 séries (linha, sinal,
    histograma) normalizadas dividindo pelo preço de fechamento.

Resultado: 10 features de entrada (5 base + 5 indicadores), tensor
`(60, 10)` — um formato de entrada diferente do `(60, 5)` das Tabelas 1-3
do artigo, por isso esta é uma geração nova e separada ("_features"), não
uma substituição de "_log"/"_tuned".

Fluxo
-----
1. Treino: para cada uma das 6 combinações (mercado × arquitetura), roda
   `train_one_model` com `feature_set='extended'` — mesmo fluxo do cenário
   1, mas com o conjunto de 10 features em vez de 5.
2. Avaliação: `evaluate_all` com `feature_set='extended'`.
3. Backtest: `backtest_all` com `feature_set='extended'`.

O que este script produz
-------------------------
- `models/{market}/{architecture}_features.keras` — os 6 modelos treinados
  com 10 features de entrada.
- `models/{market}/scaler_features.pkl` — 1 scaler por mercado (ajustado
  nas 10 colunas).
- `models/{market}/{architecture}_features_metrics.json` — métricas de
  erro por modelo.
- `results/comparison_features.csv` — tabela comparativa (métricas de erro).
- `results/backtest_comparison_features.csv` — tabela comparativa (métricas
  de negócio).
- `results/equity_curves/{market}_{architecture}_{strategy}_features.csv`.

Resultado obtido (referência — ver NOTAS.md decisão 27 para a análise
completa): acurácia direcional permaneceu ~0,49-0,50; Sharpe permaneceu
negativo nas 12 combinações, 11/12 estatisticamente significativas — o
mesmo padrão das gerações anteriores, confirmado com um conjunto de
features diferente. Quarta intervenção independente sem ganho.

Como executar
-------------
    python -m src.pipelines.05_technical_indicators

Pré-requisitos: CSVs brutos em data/dados_{WDON,WINN}_H1.csv, e a
dependência `pandas-ta` instalada (ver requirements.txt). Não depende dos
modelos de nenhuma geração anterior (treina do zero).
"""

from src.train import train_all
from src.evaluate import evaluate_all
from src.backtest import backtest_all

SUFFIX = '_features'
FEATURE_SET = 'extended'


def run():
  print("\n" + "=" * 70)
  print("CENARIO 5: Indicadores tecnicos (geracao '_features') - 10 features")
  print("=" * 70)

  print("\n--- Etapa 1/3: Treinamento dos 6 modelos (RSI, SMA, MACD + 5 base) ---")
  train_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\n--- Etapa 2/3: Avaliacao (RMSE, MAE, acuracia direcional) ---")
  evaluate_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\n--- Etapa 3/3: Backtest (Sharpe, Maximum Drawdown, VaR) ---")
  backtest_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\nConcluido. Ver results/comparison_features.csv e results/backtest_comparison_features.csv.")


if __name__ == "__main__":
  run()
