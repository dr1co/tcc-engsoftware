import os
import csv
import json

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from src.utils import create_time_windows
from src.train import (
  FEATURE_SETS,
  CLOSE_COL_IDX,
  TIMEFRAME,
  split_chronological,
  load_prepared,
  reconstruct_price,
)


def evaluate_one_model(market, architecture, suffix='_log', feature_set='base'):
  """
  Carrega o modelo e o scaler salvos para (market, architecture), reconstroi o
  conjunto de teste (em log-retorno) e o preco real correspondente, reconstroi
  o preco de fechamento previsto e calcula RMSE, MAE e acuracia direcional em
  unidades reais de preco.

  Espera modelos treinados com prepare_features (geracoes "_log"/"_tuned"/
  "_features", ver NOTAS.md decisoes 12/24/26). Nao compativel com os artefatos
  antigos (sem suffix), que foram treinados sobre preco em nivel absoluto e usam
  uma reconstrucao diferente. feature_set deve corresponder ao conjunto usado no
  treino do suffix escolhido ('base' para "_log"/"_tuned", 'extended' para
  "_features").
  """
  print(f"\n=== Avaliando {architecture.upper()} para {market} ===")

  out_dir = f"models/{market}"
  model = load_model(f"{out_dir}/{architecture}{suffix}.keras")
  scaler = joblib.load(f"{out_dir}/scaler{suffix}.pkl")

  df_returns, df_real = load_prepared(market, feature_set=feature_set)
  _, _, df_test_returns = split_chronological(df_returns)
  _, _, df_test_real = split_chronological(df_real)

  test_scaled = scaler.transform(df_test_returns.values)
  X_test, _ = create_time_windows(test_scaled, timeframe=TIMEFRAME)
  y_pred_scaled = model.predict(X_test, verbose=0).flatten()

  # dummy array para inverter a escala apenas da coluna de close (log-retorno previsto)
  n_features = len(FEATURE_SETS[feature_set])
  dummy = np.zeros((len(y_pred_scaled), n_features))
  dummy[:, CLOSE_COL_IDX] = y_pred_scaled
  predicted_log_return = scaler.inverse_transform(dummy)[:, CLOSE_COL_IDX]

  # y[i] e X_test[i] correspondem a linha (TIMEFRAME + i) do split de teste (ver create_time_windows)
  n_windows = len(X_test)
  last_close = df_test_real['close'].values[TIMEFRAME - 1: TIMEFRAME - 1 + n_windows]
  y_test = df_test_real['close'].values[TIMEFRAME: TIMEFRAME + n_windows]

  y_pred = reconstruct_price(last_close, predicted_log_return)

  rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
  mae = float(np.mean(np.abs(y_test - y_pred)))

  direction_true = np.sign(y_test - last_close)
  direction_pred = np.sign(y_pred - last_close)
  directional_accuracy = float(np.mean(direction_true == direction_pred))

  metrics = {
    'market': market,
    'architecture': architecture,
    'rmse': rmse,
    'mae': mae,
    'directional_accuracy': directional_accuracy,
  }

  print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | Acuracia direcional: {directional_accuracy:.4f}")

  metrics_path = f"{out_dir}/{architecture}{suffix}_metrics.json"
  with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
  print(f"Metricas salvas em: {metrics_path}")

  return metrics


def evaluate_all(suffix='_log', feature_set='base'):
  """Avalia os 6 modelos e salva uma tabela comparativa em results/comparison{suffix}.csv."""
  markets = ['WDON', 'WINN']
  architectures = ['cnn', 'lstm', 'hybrid']

  all_metrics = []
  for market in markets:
    for architecture in architectures:
      all_metrics.append(evaluate_one_model(market, architecture, suffix=suffix, feature_set=feature_set))

  os.makedirs('results', exist_ok=True)
  comparison_path = f'results/comparison{suffix}.csv'
  with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['market', 'architecture', 'rmse', 'mae', 'directional_accuracy'])
    writer.writeheader()
    writer.writerows(all_metrics)

  print(f"\nTabela comparativa salva em: {comparison_path}")
  return all_metrics


if __name__ == "__main__":
  evaluate_all()
