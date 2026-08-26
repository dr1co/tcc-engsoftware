import numpy as np
import pandas_ta as ta

# Parametros dos indicadores tecnicos, ajustados para o horizonte intradiario
# (candles H1) em vez dos periodos-padrao de grafico diario. Ver NOTAS.md
# (decisao 26) para o racional completo desta geracao de features ("_features").
RSI_LENGTH = 7
SMA_LENGTH = 10
MACD_FAST = 6
MACD_SLOW = 13
MACD_SIGNAL = 4

INDICATOR_COLS = ['rsi', 'sma_log_return', 'macd', 'macd_signal', 'macd_hist']


def add_technical_indicators(df):
  """
  Adiciona RSI, SMA (em log-retorno) e MACD (normalizado pelo preco) como novas
  colunas, calculadas a partir do preco real de fechamento (antes de qualquer
  transformacao em log-retorno das colunas OHLC - ver prepare_features_extended
  em src/train.py). Descarta as linhas iniciais sem indicador definido (periodo
  de aquecimento dos indicadores).

  Tratamento de escala por indicador (ver NOTAS.md decisao 26):
  - RSI: ja e um oscilador limitado a [0, 100] por construcao, mantido em nivel
    bruto (o MinMaxScaler downstream nao tem risco de extrapolacao aqui).
  - SMA: e uma serie de nivel de preco, com o mesmo risco de deriva de escala
    diagnosticado para OHLC (NOTAS.md decisao 12) - convertida para log-retorno
    da propria SMA (log(SMA_t / SMA_{t-1})) pelo mesmo motivo.
  - MACD (linha, sinal, histograma): normalizado dividindo pelo preco de
    fechamento, para permanecer comparavel entre os regimes de preco do treino
    e do teste mesmo com deriva de tendencia de longo prazo.
  """
  df = df.copy()

  df['rsi'] = ta.rsi(df['close'], length=RSI_LENGTH)

  sma = ta.sma(df['close'], length=SMA_LENGTH)
  df['sma_log_return'] = np.log(sma / sma.shift(1))

  macd_df = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
  macd_col, hist_col, signal_col = macd_df.columns  # ordem: MACD_, MACDh_, MACDs_
  df['macd'] = macd_df[macd_col] / df['close']
  df['macd_signal'] = macd_df[signal_col] / df['close']
  df['macd_hist'] = macd_df[hist_col] / df['close']

  df = df.dropna(subset=INDICATOR_COLS).reset_index(drop=True)
  return df
