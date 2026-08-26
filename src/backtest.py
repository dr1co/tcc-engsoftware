import os
import csv

import numpy as np
import joblib
from scipy import stats
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

HOLD_BARS = 4
TRADING_HOURS_PER_DAY = 9
TRADING_DAYS_PER_YEAR = 250
ANNUALIZATION_FACTOR = np.sqrt(TRADING_HOURS_PER_DAY * TRADING_DAYS_PER_YEAR)
VAR_CONFIDENCE = 0.95

# Custo fixo de round-trip (corretagem + taxas), aproximado como fracao do
# candle_range medio de cada mercado no treino. Ver NOTAS.md para o racional
# e os valores exatos usados.
TRANSACTION_COST = {
  'WDON': 2.0,
  'WINN': 40.0,
}


def _load_test_arrays(market, architecture, suffix='_log', feature_set='base'):
  """
  Carrega modelo/scaler salvos e reconstroi previsoes de preco + precos reais do periodo de teste.
  Espera modelos das geracoes "_log"/"_tuned"/"_features" (ver NOTAS.md decisoes 12/24/26) - nao
  compativel com os artefatos antigos treinados sobre preco em nivel absoluto. feature_set deve
  corresponder ao conjunto usado no treino do suffix escolhido.
  """
  out_dir = f"models/{market}"
  model = load_model(f"{out_dir}/{architecture}{suffix}.keras")
  scaler = joblib.load(f"{out_dir}/scaler{suffix}.pkl")

  df_returns, df_real = load_prepared(market, feature_set=feature_set)
  df_train_real, _, df_test_real = split_chronological(df_real)
  _, _, df_test_returns = split_chronological(df_returns)

  avg_candle_range = float((df_train_real['high'] - df_train_real['low']).mean())

  test_scaled = scaler.transform(df_test_returns.values)
  X_test, _ = create_time_windows(test_scaled, timeframe=TIMEFRAME)
  y_pred_scaled = model.predict(X_test, verbose=0).flatten()

  n_features = len(FEATURE_SETS[feature_set])
  dummy = np.zeros((len(y_pred_scaled), n_features))
  dummy[:, CLOSE_COL_IDX] = y_pred_scaled
  predicted_log_return = scaler.inverse_transform(dummy)[:, CLOSE_COL_IDX]

  # y[i]/X_test[i] correspondem a linha (TIMEFRAME + i) do split de teste (ver create_time_windows)
  n_windows = len(X_test)
  last_close = df_test_real['close'].values[TIMEFRAME - 1: TIMEFRAME - 1 + n_windows]
  y_pred = reconstruct_price(last_close, predicted_log_return)

  # OHLC reais (nao escalados) de todo o periodo de teste, na ordem cronologica,
  # para simular o caminho do preco apos cada entrada.
  test_real = df_test_real[['open', 'high', 'low', 'close']].values

  return {
    'y_pred': y_pred,
    'predicted_log_return': predicted_log_return,
    'last_close': last_close,
    'test_real': test_real,
    'avg_candle_range': avg_candle_range,
  }


def _confidence_mask(data, min_confidence_pct):
  """
  Retorna uma mascara booleana marcando quais janelas tem |predicted_log_return|
  entre os min_confidence_pct% mais "confiantes" (maior magnitude de retorno
  previsto) da propria distribuicao de previsoes deste modelo no teste.

  min_confidence_pct=100 mantem todas as janelas (equivalente a nao filtrar).
  Ver NOTAS.md (decisao 17) para o racional do filtro de confianca.
  """
  if min_confidence_pct >= 100:
    return np.ones(len(data['y_pred']), dtype=bool)

  magnitude = np.abs(data['predicted_log_return'])
  cutoff = np.percentile(magnitude, 100 - min_confidence_pct)
  return magnitude >= cutoff


def _simulate_strategy_single_bar(data, market, min_confidence_pct=100):
  """
  Estrategia A (automatizada): entra na direcao do sinal, mantem por 1 barra,
  sai no fechamento da barra seguinte. Janelas fora do percentil de confianca
  (ver _confidence_mask) sao ignoradas, isto e, o modelo "nao opera" nelas.
  """
  y_pred = data['y_pred']
  last_close = data['last_close']
  test_real = data['test_real']
  cost = TRANSACTION_COST[market]
  confident = _confidence_mask(data, min_confidence_pct)

  n_windows = len(y_pred)
  returns = []

  for i in range(n_windows):
    if not confident[i]:
      continue

    direction = np.sign(y_pred[i] - last_close[i])
    if direction == 0:
      continue

    # test_real[TIMEFRAME + i] eh a barra alvo da janela i (mesmo alinhamento de create_time_windows).
    next_bar_idx = TIMEFRAME + i
    if next_bar_idx >= len(test_real):
      break

    entry_price = last_close[i]
    exit_price = test_real[next_bar_idx, 3]  # close da proxima barra

    pnl = direction * (exit_price - entry_price) - cost
    returns.append(pnl)

  return np.array(returns)


def _simulate_strategy_multi_bar(data, market, min_confidence_pct=100):
  """
  Estrategia B (humana): entra na direcao do sinal, mantem por ate HOLD_BARS
  barras, com stop-loss/take-profit fixados no candle_range medio do treino.
  Sai antecipadamente se o preco intrabarra tocar o stop ou o alvo. Janelas
  fora do percentil de confianca (ver _confidence_mask) sao ignoradas.
  """
  y_pred = data['y_pred']
  last_close = data['last_close']
  test_real = data['test_real']
  avg_range = data['avg_candle_range']
  cost = TRANSACTION_COST[market]
  confident = _confidence_mask(data, min_confidence_pct)

  n_windows = len(y_pred)
  returns = []

  for i in range(n_windows):
    if not confident[i]:
      continue

    direction = np.sign(y_pred[i] - last_close[i])
    if direction == 0:
      continue

    entry_price = last_close[i]
    stop_price = entry_price - direction * avg_range
    target_price = entry_price + direction * avg_range

    start_idx = TIMEFRAME + i
    end_idx = min(start_idx + HOLD_BARS, len(test_real))
    if start_idx >= len(test_real):
      break

    exit_price = None
    for bar_idx in range(start_idx, end_idx):
      high = test_real[bar_idx, 1]
      low = test_real[bar_idx, 2]

      if direction > 0:
        if low <= stop_price:
          exit_price = stop_price
          break
        if high >= target_price:
          exit_price = target_price
          break
      else:
        if high >= stop_price:
          exit_price = stop_price
          break
        if low <= target_price:
          exit_price = target_price
          break

    if exit_price is None:
      exit_price = test_real[end_idx - 1, 3]  # fechamento da ultima barra da janela de holding

    pnl = direction * (exit_price - entry_price) - cost
    returns.append(pnl)

  return np.array(returns)


STRATEGIES = {
  'single_bar': _simulate_strategy_single_bar,
  'multi_bar': _simulate_strategy_multi_bar,
}


def _load_ensemble_arrays(market, suffix='_log', feature_set='base'):
  """
  Carrega as previsoes das 3 arquiteturas do mesmo mercado e monta um "data" combinado:
  y_pred = media das previsoes de preco reconstruidas das arquiteturas em concordancia
  de direcao (ver _unanimous_mask), demais campos (last_close/test_real/avg_candle_range)
  identicos entre arquiteturas do mesmo mercado (mesmo split/precos reais), reaproveitados
  de uma delas. Ver NOTAS.md decisao 18 (filtro de confianca por concordancia de ensemble).
  """
  per_arch = {
    arch: _load_test_arrays(market, arch, suffix=suffix, feature_set=feature_set)
    for arch in ['cnn', 'lstm', 'hybrid']
  }

  directions = np.stack([
    np.sign(per_arch[arch]['y_pred'] - per_arch[arch]['last_close']) for arch in per_arch
  ])  # shape (3, n_windows)
  unanimous = np.all(directions == directions[0], axis=0) & (directions[0] != 0)

  y_pred_stack = np.stack([per_arch[arch]['y_pred'] for arch in per_arch])
  y_pred_ensemble = np.mean(y_pred_stack, axis=0)

  any_arch = next(iter(per_arch.values()))
  return {
    'y_pred': y_pred_ensemble,
    'last_close': any_arch['last_close'],
    'test_real': any_arch['test_real'],
    'avg_candle_range': any_arch['avg_candle_range'],
    'unanimous_mask': unanimous,
  }


def _simulate_strategy_ensemble(data, market, strategy):
  """
  Roda a estrategia indicada (single_bar/multi_bar) usando o preco previsto medio
  do ensemble, mas so opera nas janelas em que as 3 arquiteturas concordam na
  direcao (unanimous_mask) - ver _load_ensemble_arrays e NOTAS.md decisao 18.
  """
  masked_data = dict(data)
  # reaproveita a logica de simulacao existente "escondendo" as janelas nao unanimes
  # atras do mesmo mecanismo de mascara usado pelo filtro de percentil: forcamos
  # min_confidence_pct=100 (sem corte por percentil) e sobrepomos y_pred/last_close
  # apenas nas janelas unanimes via NaN, que produzem direction=0 (sem operar).
  y_pred = np.where(data['unanimous_mask'], data['y_pred'], data['last_close'])
  masked_data['y_pred'] = y_pred

  simulate = STRATEGIES[strategy]
  return simulate(masked_data, market, min_confidence_pct=100)


def _compute_metrics(returns):
  """
  Calcula Sharpe Ratio anualizado, Maximum Drawdown, VaR historico simplificado
  e um teste-t de uma amostra (H0: media do retorno por trade = 0) a partir dos
  retornos por trade. O p-valor do teste-t indica se o retorno medio observado e
  estatisticamente distinguivel de ruido, dado o tamanho da amostra - ver NOTAS.md
  decisao 17 para o racional (necessario porque o sweep de confianca reduz
  drasticamente o numero de trades em percentis mais seletivos).
  """
  if len(returns) == 0:
    return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'var_95': 0.0, 'n_trades': 0, 'p_value': 1.0}

  mean_return = np.mean(returns)
  std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
  sharpe_ratio = float((mean_return / std_return) * ANNUALIZATION_FACTOR) if std_return > 0 else 0.0

  equity_curve = np.cumsum(returns)
  running_max = np.maximum.accumulate(equity_curve)
  drawdowns = equity_curve - running_max
  max_drawdown = float(drawdowns.min())

  var_95 = float(-np.percentile(returns, (1 - VAR_CONFIDENCE) * 100))

  if len(returns) > 1 and std_return > 0:
    _, p_value = stats.ttest_1samp(returns, popmean=0.0)
    p_value = float(p_value)
  else:
    p_value = 1.0

  return {
    'sharpe_ratio': sharpe_ratio,
    'max_drawdown': max_drawdown,
    'var_95': var_95,
    'n_trades': len(returns),
    'p_value': p_value,
  }


def backtest_one_model(market, architecture, strategy, suffix='_log', feature_set='base', min_confidence_pct=100, data=None):
  """
  Executa o backtest de uma estrategia para um par (mercado, arquitetura) e retorna as metricas de negocio.

  min_confidence_pct filtra as janelas de menor magnitude de retorno previsto (ver
  _confidence_mask); 100 (padrao) mantem o comportamento original, sem filtro.
  data permite reaproveitar previsoes ja carregadas (usado por backtest_sweep para
  nao recarregar o modelo a cada combinacao de estrategia/percentil).
  """
  print(f"\n=== Backtest [{strategy}, confianca>={min_confidence_pct}%] {architecture.upper()} para {market} ===")

  if data is None:
    data = _load_test_arrays(market, architecture, suffix=suffix, feature_set=feature_set)
  simulate = STRATEGIES[strategy]
  returns = simulate(data, market, min_confidence_pct=min_confidence_pct)

  metrics = _compute_metrics(returns)
  metrics.update({
    'market': market,
    'architecture': architecture,
    'strategy': strategy,
    'confidence_pct': min_confidence_pct,
  })

  print(
    f"Sharpe: {metrics['sharpe_ratio']:.4f} | "
    f"MaxDrawdown: {metrics['max_drawdown']:.2f} | "
    f"VaR95: {metrics['var_95']:.2f} | "
    f"Trades: {metrics['n_trades']} | "
    f"p-valor: {metrics['p_value']:.4f}"
  )

  equity_dir = "results/equity_curves"
  os.makedirs(equity_dir, exist_ok=True)
  conf_suffix = "" if min_confidence_pct == 100 else f"_conf{min_confidence_pct}"
  equity_path = f"{equity_dir}/{market}_{architecture}_{strategy}{suffix}{conf_suffix}.csv"
  with open(equity_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['trade_index', 'pnl', 'cumulative_pnl'])
    cumulative = 0.0
    for idx, pnl in enumerate(returns):
      cumulative += pnl
      writer.writerow([idx, pnl, cumulative])

  return metrics


def backtest_all(suffix='_log', feature_set='base'):
  """Executa o backtest das 12 combinacoes: 3 arquiteturas x 2 mercados x 2 estrategias (sem filtro de confianca)."""
  markets = ['WDON', 'WINN']
  architectures = ['cnn', 'lstm', 'hybrid']
  strategies = ['single_bar', 'multi_bar']

  all_metrics = []
  for market in markets:
    for architecture in architectures:
      for strategy in strategies:
        all_metrics.append(backtest_one_model(market, architecture, strategy, suffix=suffix, feature_set=feature_set))

  os.makedirs('results', exist_ok=True)
  comparison_path = f'results/backtest_comparison{suffix}.csv'
  with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['market', 'architecture', 'strategy', 'sharpe_ratio', 'max_drawdown', 'var_95', 'n_trades', 'p_value']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({k: v for k, v in m.items() if k != 'confidence_pct'} for m in all_metrics)

  print(f"\nTabela comparativa de backtest salva em: {comparison_path}")
  return all_metrics


CONFIDENCE_GRID = [100, 75, 50, 25, 10, 5]


def backtest_sweep(suffix='_log', feature_set='base', confidence_grid=CONFIDENCE_GRID):
  """
  Executa o backtest das 12 combinacoes (mercado x arquitetura x estrategia) em cada
  percentil de confianca de confidence_grid, filtrando por |predicted_log_return|
  (ver _confidence_mask e NOTAS.md decisao 17). Reaproveita as previsoes carregadas
  por (mercado, arquitetura) entre estrategias e percentis, evitando recarregar o
  modelo a cada combinacao. Salva o resultado consolidado em results/backtest_sweep{suffix}.csv.
  """
  markets = ['WDON', 'WINN']
  architectures = ['cnn', 'lstm', 'hybrid']
  strategies = ['single_bar', 'multi_bar']

  all_metrics = []
  for market in markets:
    for architecture in architectures:
      data = _load_test_arrays(market, architecture, suffix=suffix, feature_set=feature_set)
      for strategy in strategies:
        for pct in confidence_grid:
          all_metrics.append(
            backtest_one_model(market, architecture, strategy, suffix=suffix, feature_set=feature_set, min_confidence_pct=pct, data=data)
          )

  os.makedirs('results', exist_ok=True)
  sweep_path = f'results/backtest_sweep{suffix}.csv'
  with open(sweep_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['market', 'architecture', 'strategy', 'confidence_pct', 'sharpe_ratio', 'max_drawdown', 'var_95', 'n_trades', 'p_value']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_metrics)

  print(f"\nTabela do sweep de confianca salva em: {sweep_path}")
  return all_metrics


def backtest_ensemble_one(market, strategy, suffix='_log', feature_set='base'):
  """
  Executa o backtest da estrategia indicada usando o filtro de concordancia de
  ensemble (so opera quando CNN, LSTM e Hibrido concordam na direcao; preco
  previsto = media das 3). Ver NOTAS.md decisao 18.
  """
  print(f"\n=== Backtest ensemble [{strategy}, unanimidade 3/3] para {market} ===")

  data = _load_ensemble_arrays(market, suffix=suffix, feature_set=feature_set)
  returns = _simulate_strategy_ensemble(data, market, strategy)

  metrics = _compute_metrics(returns)
  metrics.update({'market': market, 'architecture': 'ensemble', 'strategy': strategy})

  print(
    f"Sharpe: {metrics['sharpe_ratio']:.4f} | "
    f"MaxDrawdown: {metrics['max_drawdown']:.2f} | "
    f"VaR95: {metrics['var_95']:.2f} | "
    f"Trades: {metrics['n_trades']} | "
    f"p-valor: {metrics['p_value']:.4f} | "
    f"Janelas unanimes: {data['unanimous_mask'].sum()}/{len(data['unanimous_mask'])}"
  )

  equity_dir = "results/equity_curves"
  os.makedirs(equity_dir, exist_ok=True)
  equity_path = f"{equity_dir}/{market}_ensemble_{strategy}{suffix}.csv"
  with open(equity_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['trade_index', 'pnl', 'cumulative_pnl'])
    cumulative = 0.0
    for idx, pnl in enumerate(returns):
      cumulative += pnl
      writer.writerow([idx, pnl, cumulative])

  return metrics


def backtest_ensemble_all(suffix='_log', feature_set='base'):
  """Executa o backtest do filtro de concordancia de ensemble para os 2 mercados x 2 estrategias."""
  markets = ['WDON', 'WINN']
  strategies = ['single_bar', 'multi_bar']

  all_metrics = []
  for market in markets:
    for strategy in strategies:
      all_metrics.append(backtest_ensemble_one(market, strategy, suffix=suffix, feature_set=feature_set))

  os.makedirs('results', exist_ok=True)
  ensemble_path = f'results/backtest_ensemble{suffix}.csv'
  with open(ensemble_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['market', 'architecture', 'strategy', 'sharpe_ratio', 'max_drawdown', 'var_95', 'n_trades', 'p_value']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_metrics)

  print(f"\nTabela do backtest de ensemble salva em: {ensemble_path}")
  return all_metrics


if __name__ == "__main__":
  backtest_all()
