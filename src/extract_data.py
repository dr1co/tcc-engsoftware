import pandas as pd
from datetime import datetime
from mt5linux import MetaTrader5

mt5 = MetaTrader5(host='localhost', port=18812)

def conectar_mt5_ponte():
  """Conecta ao servidor da ponte mt5linux rodando dentro do Bottles."""
  print("Conectando à ponte MT5 no Bottles...")

  if not mt5.initialize():
    print(f"Erro ao conectar na ponte MT5: {mt5.last_error()}")
    return False
  
  info_terminal = mt5.terminal_info()
  print(f"Conectado com sucesso ao MT5! Terminal: {info_terminal.name}")
  return True

def extrair_historico_candles(ativo, timeframe, data_inicio, data_fim):
  """
  Solicita o histórico de candles do ativo selecionado.
  """

  if not mt5.symbol_select(ativo, True):
    print(f"Aviso: Não foi possível selecionar o ativo '{ativo}'. Verifique se o nome está correto no MT5.")
    return None

  print(f"Extraindo dados de {ativo} [{data_inicio.strftime('%Y-%m-%d')} até {data_fim.strftime('%Y-%m-%d')}]...")
  
  rates = mt5.copy_rates_range(ativo, timeframe, data_inicio, data_fim)

  if rates is None or len(rates) == 0:
    print(f"Falha ao obter dados para {ativo}. Verifique se o ativo possui histórico no MT5.")
    return None

  df = pd.DataFrame(rates)

  df['time'] = pd.to_datetime(df['time'], unit='s')
  
  print(f"Sucesso! {len(df)} registros/candles extraídos.")
  return df

def pipeline():
  ATIVO = "WDO$N" 
  TIMEFRAME = mt5.TIMEFRAME_H1      
  DATA_INICIO = datetime(2021, 1, 1)
  DATA_FIM = datetime.now()
  NOME_ARQUIVO_SAIDA = f"dados_{ATIVO}_{DATA_INICIO.strftime('%Y%m%d')}.csv"

  if not conectar_mt5_ponte():
    return

  try:
    df_dados = extrair_historico_candles(ATIVO, TIMEFRAME, DATA_INICIO, DATA_FIM)

    if df_dados is not None:
      # Seleção e renomeação de colunas relevantes para o pipeline de DL
      # Colunas originais: time, open, high, low, close, tick_volume, spread, real_volume
      cols_interesse = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']
      df_final = df_dados[cols_interesse]

      # 4. Salva o CSV na pasta local do Linux
      df_final.to_csv(NOME_ARQUIVO_SAIDA, index=False)
      print(f"Arquivo salvo com sucesso em: {NOME_ARQUIVO_SAIDA}")

      # Visualização rápida das primeiras linhas
      print("\n--- Amostra dos Dados Extraídos ---")
      print(df_final.head())

  finally:
    # 5. Encerra a conexão com a ponte
    mt5.shutdown()
    print("\nConexão com MT5 encerrada.")
