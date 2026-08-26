"""
Cenário 6 (final) — Filtro de confiança por magnitude reaplicado sobre as
gerações "_tuned" e "_features", com verificação de comparações múltiplas.

Contexto (NOTAS.md decisão 28)
--------------------------------
Com quatro intervenções independentes (cenários 2, 3, 4, 5) sem produzir
uma vantagem estatisticamente confiável, este cenário testa uma última
combinação: será que um modelo mais bem calibrado (tuning, cenário 4) ou
com mais informação de entrada (indicadores técnicos, cenário 5), quando
TAMBÉM filtrado por confiança (mesmo método do cenário 2), produz uma
vantagem que resista a um teste de significância?

O filtro de confiança por magnitude (cenário 2) foi escolhido para esta
combinação final por ser, das intervenções testadas, a única que já havia
produzido pelo menos um Sharpe positivo (ainda que não significativo) —
ver NOTAS.md decisão 17.

Este script roda a mesma grade de percentis do cenário 2
(`backtest_sweep`), mas sobre as gerações "_tuned" (`feature_set='base'`)
e "_features" (`feature_set='extended'`), e então agrega os resultados das
3 gerações ("_log", já produzido pelo cenário 2; "_tuned"; "_features")
para uma verificação estatística que os cenários 2-5, isolados, não
faziam: correção para comparações múltiplas.

Por que a correção é necessária: ao todo, rodam-se 72 testes de hipótese
por geração (12 combinações mercado×arquitetura×estratégia × 6 percentis).
Com 3 gerações testadas com o mesmo método, são 216 testes no total. A um
limiar de significância de 5% (p < 0,05), o número ESPERADO de falsos
positivos por puro acaso, mesmo que nenhuma combinação tivesse qualquer
vantagem real, é `216 × 0,05 ≈ 10,8`. Encontrar um punhado de combinações
com `p < 0,05` não é, por si só, evidência de uma vantagem real — pode ser
exatamente o que o acaso produziria ao testar 216 hipóteses. A correção de
Bonferroni (dividir o limiar de significância pelo número de testes:
`0,05 / 216 ≈ 0,000231`) é o padrão simples e conservador para esse
problema.

Fluxo
-----
1. Pré-requisito: os modelos das gerações "_tuned" (cenário 4) e
   "_features" (cenário 5) devem já existir. O sweep da geração "_log"
   (cenário 2) também deve já ter sido rodado, já que este script reusa o
   arquivo `results/backtest_sweep_log.csv` em vez de recalculá-lo.
2. Roda `backtest_sweep` sobre "_tuned" (`results/backtest_sweep_tuned.csv`).
3. Roda `backtest_sweep` sobre "_features" (`results/backtest_sweep_features.csv`).
4. Agrega os 3 arquivos de sweep (log/tuned/features), conta quantas
   combinações são simultaneamente positivas E significativas (p < 0,05)
   em cada geração, e aplica a correção de Bonferroni sobre o total de 216
   testes para determinar se alguma sobrevive.

O que este script produz
-------------------------
- `results/backtest_sweep_tuned.csv` — sweep de confiança sobre "_tuned"
  (72 linhas).
- `results/backtest_sweep_features.csv` — sweep de confiança sobre
  "_features" (72 linhas).
- Impressão no console de um resumo comparativo das 3 gerações (contagem
  de combinações positivas/significativas) e da verificação de Bonferroni
  sobre qualquer combinação que pareça, à primeira vista, promissora.

Resultado obtido (referência — ver NOTAS.md decisão 28 para a análise
completa): 3 combinações na geração "_features" (todas no percentil mais
extremo, 5% de confiança, 88 trades) tiveram Sharpe positivo (+10,5 a
+12,3) E p < 0,05 (0,017 a 0,041) — a primeira vez que qualquer geração
produziu um resultado "significativo" na direção positiva. Nenhuma das 3,
porém, sobrevive à correção de Bonferroni (limiar corrigido ≈ 0,000231) —
consistente com ruído de comparações múltiplas, não com uma vantagem real.
Conclusão final do experimento: nenhuma combinação de arquitetura,
hiperparâmetros, conjunto de features ou regra de seleção de trades
testada produziu uma vantagem estatisticamente defensável.

Como executar
-------------
    python -m src.pipelines.06_combined_final_check

Pré-requisitos: modelos das geracoes "_tuned" (cenario 4) e "_features"
(cenario 5) ja treinados, e results/backtest_sweep_log.csv ja gerado
(cenario 2).
"""

import os
import pandas as pd

from src.backtest import backtest_sweep, CONFIDENCE_GRID

LOG_SWEEP_PATH = 'results/backtest_sweep_log.csv'
TUNED_SWEEP_PATH = 'results/backtest_sweep_tuned.csv'
FEATURES_SWEEP_PATH = 'results/backtest_sweep_features.csv'


def _summarize(df, label):
  total = len(df)
  n_pos = int((df['sharpe_ratio'] > 0).sum())
  n_sig = int((df['p_value'] < 0.05).sum())
  n_sig_pos = int(((df['p_value'] < 0.05) & (df['sharpe_ratio'] > 0)).sum())
  best = df.loc[df['sharpe_ratio'].idxmax()]
  print(f"  {label:12s} | combinacoes: {total:3d} | Sharpe>0: {n_pos:2d} | "
        f"significativas (p<0.05): {n_sig:2d} | significativas E positivas: {n_sig_pos}")
  print(f"               melhor Sharpe: {best['sharpe_ratio']:.4f} "
        f"({best['market']}/{best['architecture']}/{best['strategy']}, "
        f"confianca={best['confidence_pct']}%, trades={int(best['n_trades'])}, p={best['p_value']:.4f})")
  return n_sig_pos


def run():
  print("\n" + "=" * 70)
  print("CENARIO 6 (final): Filtro de confianca sobre '_tuned' e '_features'")
  print("+ verificacao de comparacoes multiplas (Bonferroni)")
  print("=" * 70)

  print("\n--- Etapa 1/2: Sweep de confianca sobre a geracao '_tuned' ---")
  backtest_sweep(suffix='_tuned', feature_set='base', confidence_grid=CONFIDENCE_GRID)

  print("\n--- Etapa 2/2: Sweep de confianca sobre a geracao '_features' ---")
  backtest_sweep(suffix='_features', feature_set='extended', confidence_grid=CONFIDENCE_GRID)

  print("\n" + "-" * 70)
  print("Resumo agregado das 3 geracoes (log = cenario 2, ja deve existir):")
  print("-" * 70)

  if not os.path.exists(LOG_SWEEP_PATH):
    print(f"\nAVISO: {LOG_SWEEP_PATH} nao encontrado - rode o cenario 2 antes "
          "para incluir a geracao '_log' na comparacao agregada.")
    return

  df_log = pd.read_csv(LOG_SWEEP_PATH)
  df_tuned = pd.read_csv(TUNED_SWEEP_PATH)
  df_features = pd.read_csv(FEATURES_SWEEP_PATH)

  n_sig_pos_log = _summarize(df_log, '_log')
  n_sig_pos_tuned = _summarize(df_tuned, '_tuned')
  n_sig_pos_features = _summarize(df_features, '_features')

  total_tests = len(df_log) + len(df_tuned) + len(df_features)
  expected_false_positives = total_tests * 0.05
  bonferroni_alpha = 0.05 / total_tests

  print("\n" + "-" * 70)
  print(f"Total de testes de hipotese nas 3 geracoes: {total_tests}")
  print(f"Falsos positivos esperados por acaso a p<0.05: {expected_false_positives:.1f}")
  print(f"Limiar de significancia corrigido (Bonferroni): {bonferroni_alpha:.6f}")
  print("-" * 70)

  all_sig_pos = pd.concat([
    df_log[(df_log['p_value'] < 0.05) & (df_log['sharpe_ratio'] > 0)].assign(generation='_log'),
    df_tuned[(df_tuned['p_value'] < 0.05) & (df_tuned['sharpe_ratio'] > 0)].assign(generation='_tuned'),
    df_features[(df_features['p_value'] < 0.05) & (df_features['sharpe_ratio'] > 0)].assign(generation='_features'),
  ])

  if len(all_sig_pos) == 0:
    print("\nNenhuma combinacao teve Sharpe positivo E p<0.05 em nenhuma geracao.")
    print("CONCLUSAO: nenhuma vantagem estatisticamente confiavel foi encontrada.")
  else:
    survivors = all_sig_pos[all_sig_pos['p_value'] < bonferroni_alpha]
    print(f"\n{len(all_sig_pos)} combinacao(oes) com Sharpe>0 e p<0.05 (antes da correcao):")
    print(all_sig_pos[['generation', 'market', 'architecture', 'strategy', 'confidence_pct', 'sharpe_ratio', 'n_trades', 'p_value']].to_string(index=False))
    print(f"\nDessas, {len(survivors)} sobrevivem a correcao de Bonferroni (p < {bonferroni_alpha:.6f}).")
    if len(survivors) == 0:
      print("CONCLUSAO: os resultados aparentemente positivos sao consistentes com ruido")
      print("de comparacoes multiplas, nao com uma vantagem direcional real.")
    else:
      print("ATENCAO: ao menos uma combinacao sobrevive a correcao - investigar mais a fundo")
      print("antes de descartar como ruido.")


if __name__ == "__main__":
  run()
