import numpy as np

def create_time_windows(dados, timeframe=60):
  """
  Transforma uma tabela 2D [registros x colunas] em matrizes 3D [janelas x tempo x colunas]
  """
  X = []
  y = []
  
  for i in range(len(dados) - timeframe):
    janela = dados[i:(i + timeframe), :]
    X.append(janela)
    
    alvo = dados[i + timeframe, 3] 
    y.append(alvo)
        
  return np.array(X), np.array(y)
