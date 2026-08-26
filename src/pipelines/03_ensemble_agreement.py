"""
Cenário 3 — Filtro de confiança por concordância de ensemble (unanimidade
entre CNN, LSTM e Híbrido), aplicado sobre a geração "_log".

Contexto (NOTAS.md decisão 18)
--------------------------------
Segunda tentativa de seletividade de trades, usando uma medida de confiança
diferente da magnitude (cenário 2): em vez de olhar para uma única
previsão isoladamente, este cenário só opera nas janelas em que as 3
arquiteturas do mesmo mercado (treinadas de forma independente, mesmos
dados) concordam no sinal de `(preço_previsto - último_fechamento)` —
unanimidade 3/3, não maioria 2/3. Quando há concordância, o preço previsto
usado para a operação é a média dos 3 preços reconstruídos.

Intuição testada: se 3 modelos independentes concordam, isso seria um
sinal de convicção mais robusto que o ruído específico de uma arquitetura
isolada.

Fluxo
-----
1. Pré-requisito: os 6 modelos da geração "_log" devem já existir (rode o
   cenário 1 antes) — este cenário precisa dos 3 modelos de cada mercado
   simultaneamente (CNN, LSTM, Híbrido), não de um por vez.
2. Para cada mercado, carrega as previsões dos 3 modelos, calcula a máscara
   de unanimidade e a previsão média do ensemble.
3. Roda as 2 estratégias (single_bar, multi_bar) para os 2 mercados (4
   combinações), operando apenas nas janelas unânimes.

O que este script produz
-------------------------
- `results/backtest_ensemble_log.csv` — tabela com 4 linhas (2 mercados × 2
  estratégias), mesmas colunas de métricas do cenário 2 (sem confidence_pct,
  já que não há grade de percentis aqui — é um filtro binário, unânime ou não).
- `results/equity_curves/{market}_ensemble_{strategy}_log.csv` — curva de
  equity por combinação (4 arquivos).

Resultado obtido (referência — ver NOTAS.md decisão 18 para a análise
completa): as 4 combinações tiveram Sharpe negativo (nenhuma reverteu para
positivo), 3 das 4 estatisticamente significativas (p < 0,001) — reforça,
não contradiz, o achado do cenário 2. A taxa de concordância entre
arquiteturas foi relativamente alta (60-82% das janelas), mas concordância
não é o mesmo que acerto: é esperado que modelos treinados nos mesmos
dados capturem tendências (e vieses) parecidos mesmo sem uma vantagem
preditiva real individual.

Como executar
-------------
    python -m src.pipelines.03_ensemble_agreement

Pré-requisitos: os 3 modelos (CNN, LSTM, Hibrido) de cada mercado, da
geracao "_log", ja treinados (cenario 1).
"""

from src.backtest import backtest_ensemble_all

SUFFIX = '_log'
FEATURE_SET = 'base'


def run():
  print("\n" + "=" * 70)
  print("CENARIO 3: Filtro de confianca por concordancia de ensemble (geracao '_log')")
  print("=" * 70)

  backtest_ensemble_all(suffix=SUFFIX, feature_set=FEATURE_SET)

  print("\nConcluido. Ver results/backtest_ensemble_log.csv.")


if __name__ == "__main__":
  run()
