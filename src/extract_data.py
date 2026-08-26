import pandas as pd
from datetime import datetime
from mt5linux import MetaTrader5

mt5 = MetaTrader5(host='localhost', port=18812)

def connect_mt5():
  """Conecta ao servidor da ponte mt5linux rodando dentro do Bottles."""
  print("Conectando à ponte MT5 no Bottles...")

  if not mt5.initialize():
    print(f"Erro ao conectar na ponte MT5: {mt5.last_error()}")
    return False
  
  info_terminal = mt5.terminal_info()
  print(f"Conectado com sucesso ao MT5! Terminal: {info_terminal.name}")
  return True

def extract_candle_history(market, timeframe, start_date, end_date):
  """
  Solicita o histórico de candles do ativo selecionado.
  """

  if not mt5.symbol_select(market, True):
    print(f"Aviso: Não foi possível selecionar o ativo '{market}'. Verifique se o nome está correto no MT5.")
    return None

  print(f"Extraindo dados de {market} [{start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}]...")
  
  rates = mt5.copy_rates_range(market, timeframe, start_date, end_date)

  if rates is None or len(rates) == 0:
    print(f"Falha ao obter dados para {market}. Verifique se o ativo possui histórico no MT5.")
    return None

  df = pd.DataFrame(rates)

  df['time'] = pd.to_datetime(df['time'], unit='s')
  
  print(f"Sucesso! {len(df)} registros/candles extraídos.")
  return df

def extract_market(market: str):
  MARKET = market
  TIMEFRAME = mt5.TIMEFRAME_H1
  START_DATE = datetime(2021, 1, 1)
  END_DATE = datetime.now()
  OUT_FILENAME = f"dados_{MARKET}_{START_DATE.strftime('%Y%m%d')}.csv"

  if not connect_mt5():
    return

  try:
    df_data = extract_candle_history(MARKET, TIMEFRAME, START_DATE, END_DATE)

    if df_data is not None:
      # Seleção e renomeação de colunas relevantes para o pipeline de DL
      # Colunas originais: time, open, high, low, close, tick_volume, spread, real_volume
      cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']
      df_final = df_data[cols]

      # 4. Salva o CSV na pasta local do Linux
      df_final.to_csv(OUT_FILENAME, index=False)
      print(f"Arquivo salvo com sucesso em: {OUT_FILENAME}")

      # Visualização rápida das primeiras linhas
      print("\n--- Amostra dos Dados Extraídos ---")
      print(df_final.head())

  finally:
    # 5. Encerra a conexão com a ponte
    mt5.shutdown()
    print("\nConexão com MT5 encerrada.")
