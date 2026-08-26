"""
Cenário 2 — Filtro de confiança por magnitude do log-retorno previsto,
aplicado sobre a geração "_log".

Contexto (NOTAS.md decisão 17)
--------------------------------
O baseline (cenário 1) opera em praticamente todas as janelas do conjunto
de teste (~1750 trades por mercado) — um trader real seria mais seletivo,
só operando quando "mais confiante". Este cenário testa se restringir os
trades às previsões de maior magnitude de `|log-retorno previsto|` (usado
como proxy de confiança do próprio modelo) melhora o resultado do backtest.

Para cada modelo, o corte de percentil é calculado a partir da própria
distribuição de previsões daquele modelo no teste (não um valor absoluto
fixo, nem compartilhado entre arquiteturas) — ver `_confidence_mask` em
`src/backtest.py`.

Grade de percentis testada: 100% (baseline, sem filtro), 75%, 50%, 25%,
10%, 5% das previsões de maior magnitude mantidas.

IMPORTANTE — leitura dos resultados: Sharpe Ratio sozinho é enganoso
quando o número de trades cai muito (de ~1750 para ~88 no percentil de
5%) — um retorno médio positivo em amostra pequena pode ser puro ruído
estatístico. Por isso `backtest_one_model`/`backtest_sweep` também
computam um teste-t de uma amostra (H0: retorno médio = 0) e reportam o
p-valor ao lado do Sharpe. Nenhuma conclusão deve ser tirada olhando só
para o Sharpe.

Fluxo
-----
1. Pré-requisito: os 6 modelos da geração "_log" devem já existir em
   `models/{market}/{architecture}_log.keras` (rode o cenário 1 antes).
2. Para cada uma das 12 combinações mercado×arquitetura×estratégia, roda o
   backtest em cada um dos 6 percentis da grade, reaproveitando as
   previsões já carregadas (sem recarregar o modelo a cada percentil).

O que este script produz
-------------------------
- `results/backtest_sweep_log.csv` — tabela com 72 linhas (12 combinações
  × 6 percentis), colunas: market, architecture, strategy, confidence_pct,
  sharpe_ratio, max_drawdown, var_95, n_trades, p_value.
- `results/equity_curves/{market}_{architecture}_{strategy}_log_conf{pct}.csv`
  — curva de equity por combinação×percentil (exceto pct=100, que reusa o
  arquivo já gerado pelo cenário 1, sem sufixo "_confXX").

Resultado obtido (referência — ver NOTAS.md decisão 17 para a análise
completa): das 72 combinações, 7 tiveram Sharpe positivo, mas todas com
p-valor > 0,4 (estatisticamente indistinguível de ruído). Das 41
combinações estatisticamente significativas (p < 0,05), todas tiveram
Sharpe negativo. Conclusão: filtrar por magnitude do log-retorno previsto
não produziu uma vantagem confiável nesta geração de modelos.

Como executar
-------------
    python -m src.pipelines.02_confidence_filter

Pré-requisitos: modelos da geração "_log" já treinados (cenário 1).
"""

from src.backtest import backtest_sweep, CONFIDENCE_GRID

SUFFIX = '_log'
FEATURE_SET = 'base'


def run():
  print("\n" + "=" * 70)
  print("CENARIO 2: Filtro de confianca por magnitude (geracao '_log')")
  print(f"Grade de percentis: {CONFIDENCE_GRID}")
  print("=" * 70)

  backtest_sweep(suffix=SUFFIX, feature_set=FEATURE_SET, confidence_grid=CONFIDENCE_GRID)

  print("\nConcluido. Ver results/backtest_sweep_log.csv.")
  print("Lembrete: verifique o p-valor antes de considerar qualquer Sharpe positivo como vantagem real.")


if __name__ == "__main__":
  run()
