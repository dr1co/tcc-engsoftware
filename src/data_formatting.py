import os
import pandas as pd
import numpy as np

def sanitizar_dataframe(filepath_in, filepath_out):
  print(f"\n--- Processando: {filepath_in} ---")
  
  # 1. Carregamento dos dados
  df = pd.read_csv(filepath_in)
  print(f"Candles originais: {len(df)}")
  
  # 2. Ordenação e Conversão de Tipos
  df['time'] = pd.to_datetime(df['time'])
  df = df.sort_values('time').reset_index(drop=True)
  
  # 3. Remoção de Duplicatas de Timestamps
  duplicados = df.duplicated(subset=['time']).sum()
  if duplicados > 0:
    print(f"Removendo {duplicados} registros duplicados...")
    df = df.drop_duplicates(subset=['time'], keep='first').reset_index(drop=True)
      
  # 4. Tratamento de Valores Ausentes (NaNs/Nulos)
  if df.isnull().sum().sum() > 0:
    print("Tratando valores ausentes...")
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill().bfill()
    df[['tick_volume', 'real_volume']] = df[['tick_volume', 'real_volume']].fillna(0)
      
  # 5. Validação de Inconsistências de Cotação (High >= Low/Open/Close, etc.)
  # Remove candles corrompidos onde a mínima é maior que a máxima ou preços <= 0
  invalidos = (
    (df['high'] < df['low']) | 
    (df['high'] < df['open']) | 
    (df['high'] < df['close']) | 
    (df['low'] > df['open']) | 
    (df['low'] > df['close']) |
    (df['close'] <= 0)
  )
  if invalidos.sum() > 0:
    print(f"Removendo {invalidos.sum()} candles com inconsistências de preços...")
    df = df[~invalidos].reset_index(drop=True)
      
  # 6. Engenharia de Features Iniciais para Séries Temporais
  # Adiciona retornos logarítmicos (estacionariedade essencial para redes neurais)
  df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
  
  # Range relativo do candle (volatilidade normalizada)
  df['candle_range'] = (df['high'] - df['low']) / df['close']
  
  # 7. Salvar dataset limpo
  os.makedirs(os.path.dirname(filepath_out), exist_ok=True)
  df.to_csv(filepath_out, index=False)
  print(f"Sanitização concluída! Candles válidos: {len(df)}")
  print(f"Arquivo sanitizado salvo em: {filepath_out}")
  return df
