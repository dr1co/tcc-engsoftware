import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data_formatting import sanitize_df
from src.utils import create_time_windows
from src.models import create_model_cnn, create_model_lstm, create_model_hybrid
from src.features import add_technical_indicators, INDICATOR_COLS
from tensorflow.keras.callbacks import EarlyStopping

# Colunas de preco (OHLC) sao transformadas em log-retorno antes de entrar no
# pipeline (ver prepare_features); real_volume permanece em nivel bruto.
# A ordem e o numero de colunas (5) sao mantidos identicos aos anteriores para
# nao alterar o formato de entrada (60, 5) documentado no TCC (Tabelas 1-3).
PRICE_COLS = ['open', 'high', 'low', 'close']
FEATURE_COLS = ['open', 'high', 'low', 'close', 'real_volume']

# Geracao "_features" (NOTAS.md decisao 26): adiciona indicadores tecnicos
# (RSI, SMA em log-retorno, MACD normalizado pelo preco) as 5 features base,
# totalizando 10 - input shape (60, 10), documentado em tabela propria, nao
# nas Tabelas 1-3 do TCC (que descrevem especificamente a arquitetura base).
FEATURE_COLS_EXTENDED = FEATURE_COLS + INDICATOR_COLS

FEATURE_SETS = {
  'base': FEATURE_COLS,
  'extended': FEATURE_COLS_EXTENDED,
}

CLOSE_COL_IDX = FEATURE_COLS.index('close')  # mesma posicao (3) em ambos os conjuntos, close nao muda de indice
TIMEFRAME = 60
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

MODEL_BUILDERS = {
  'cnn': create_model_cnn,
  'lstm': create_model_lstm,
  'hybrid': create_model_hybrid,
}


def prepare_features(df):
  """
  Converte open/high/low/close em log-retorno (log(X_t / X_{t-1})), cada coluna
  em relacao ao seu proprio valor anterior, e mantem real_volume (e quaisquer
  colunas de indicadores tecnicos ja calculadas, ver prepare_features_extended)
  em nivel bruto. A primeira linha (sem valor anterior) e descartada.

  Motivo: Min-Max Scaler sobre preco em nivel absoluto assume que o intervalo
  de precos do teste fica contido no intervalo observado no treino. Para series
  fortemente tendenciosas (ex.: WIN em alta de longo prazo), o preco no periodo
  de teste pode ficar muito acima do maximo visto no treino, levando o modelo a
  extrapolar fora do dominio em que foi treinado. log_return e estacionario -
  sua distribuicao nao se desloca com o nivel absoluto do preco - entao o
  problema deixa de existir independente de quanto o preco tenha subido desde
  o treino. Ver NOTAS.md (decisao 12) para o diagnostico completo.
  """
  df = df.copy()
  for col in PRICE_COLS:
    df[col] = np.log(df[col] / df[col].shift(1))
  df = df.iloc[1:].reset_index(drop=True)
  return df


def reconstruct_price(last_real_close, predicted_log_return):
  """Reconstroi o preco de fechamento previsto a partir do ultimo fechamento real conhecido e do log-retorno previsto."""
  return last_real_close * np.exp(predicted_log_return)


def split_chronological(df):
  """Divide o DataFrame em treino/validação/teste preservando a ordem cronológica."""
  n = len(df)
  train_end = int(n * TRAIN_FRAC)
  val_end = train_end + int(n * VAL_FRAC)

  df_train = df.iloc[:train_end]
  df_val = df.iloc[train_end:val_end]
  df_test = df.iloc[val_end:]

  return df_train, df_val, df_test


def load_prepared(market, feature_set='base'):
  """
  Carrega o CSV bruto de um mercado, aplica sanitizacao e o pipeline de features
  do conjunto indicado ('base': 5 features OHLC+volume em log-retorno, ver
  decisao 12; 'extended': as 5 base + 5 indicadores tecnicos, ver
  src/features.py e NOTAS.md decisao 26), e retorna tanto o DataFrame pronto
  para o modelo quanto o DataFrame de precos reais alinhado linha a linha com
  ele (para reconstrucao de preco e simulacao de trades).
  """
  filepath_in = f"data/dados_{market}_H1.csv"
  df_sanitized = sanitize_df(filepath_in)

  if feature_set == 'extended':
    # add_technical_indicators descarta o periodo de aquecimento dos indicadores
    # (mais que 1 linha, ao contrario de prepare_features) - a selecao de colunas
    # e o descarte de linhas precisam acontecer nessa ordem para real_volume e as
    # colunas de indicadores ficarem disponiveis antes de prepare_features rodar.
    df_with_indicators = add_technical_indicators(df_sanitized)
    df_raw = df_with_indicators[FEATURE_COLS_EXTENDED]
  else:
    df_raw = df_sanitized[FEATURE_COLS]

  # prepare_features descarta a primeira linha (sem retorno definido); o DataFrame
  # de precos reais precisa do mesmo descarte para os indices continuarem alinhados.
  df_real = df_raw.iloc[1:].reset_index(drop=True)
  df_returns = prepare_features(df_raw)

  return df_returns, df_real


def train_one_model(market, architecture, epochs=200, batch_size=32, patience=12, suffix='_log', feature_set='base'):
  """
  Executa o pipeline completo para um par (mercado, arquitetura):
  sanitizacao -> features -> split cronologico -> escalonamento -> janelamento -> treino -> salvamento.

  O parametro suffix distingue geracoes de modelos: a original (preco em nivel
  absoluto, sem suffix), "_log" (features em log-retorno, decisao 12) e
  "_tuned"/"_features" (variacoes subsequentes, ver NOTAS.md). feature_set
  seleciona o conjunto de colunas usado ('base': 5 features; 'extended': 5 +
  indicadores tecnicos, ver src/features.py e decisao 26) e deve corresponder
  ao suffix escolhido para nao misturar geracoes incompativeis.
  """
  print(f"\n=== Treinando {architecture.upper()} para {market} ===")

  df, _ = load_prepared(market, feature_set=feature_set)
  df_train, df_val, df_test = split_chronological(df)

  scaler = MinMaxScaler(feature_range=(0, 1))
  train_scaled = scaler.fit_transform(df_train.values)
  val_scaled = scaler.transform(df_val.values)
  test_scaled = scaler.transform(df_test.values)

  X_train, y_train = create_time_windows(train_scaled, timeframe=TIMEFRAME)
  X_val, y_val = create_time_windows(val_scaled, timeframe=TIMEFRAME)
  X_test, y_test = create_time_windows(test_scaled, timeframe=TIMEFRAME)

  build_model = MODEL_BUILDERS[architecture]
  model = build_model(timeframe=TIMEFRAME, num_features=len(FEATURE_SETS[feature_set]))

  early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True,
  )

  history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=2,
  )

  epochs_run = len(history.history['loss'])
  print(f"Treino finalizado em {epochs_run} epocas (max={epochs}, patience={patience}).")

  out_dir = f"models/{market}"
  os.makedirs(out_dir, exist_ok=True)

  model_path = f"{out_dir}/{architecture}{suffix}.keras"
  model.save(model_path)
  scaler_path = f"{out_dir}/scaler{suffix}.pkl"
  if not os.path.exists(scaler_path):
    joblib.dump(scaler, scaler_path)

  print(f"Modelo salvo em: {model_path}")

  return model, scaler, (X_test, y_test)


def train_all(suffix='_log', feature_set='base'):
  """Treina os 6 modelos: 3 arquiteturas x 2 mercados (WDON, WINN)."""
  markets = ['WDON', 'WINN']
  architectures = ['cnn', 'lstm', 'hybrid']

  for market in markets:
    for architecture in architectures:
      train_one_model(market, architecture, suffix=suffix, feature_set=feature_set)
