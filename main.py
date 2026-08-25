import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.extract_data import pipeline

import numpy as np

def create_time_windows(dados, janela_tempo=60):
  """
  Transforma uma tabela 2D [registros x colunas] em matrizes 3D [janelas x tempo x colunas]
  """
  X = []
  y = []
  
  for i in range(len(dados) - janela_tempo):
    janela = dados[i:(i + janela_tempo), :]
    X.append(janela)
    
    alvo = dados[i + janela_tempo, 3] 
    y.append(alvo)
        
  return np.array(X), np.array(y)

def create_model_cnn(janela_tempo=60, num_features=5):
  model = models.Sequential()
  
  # Mesma estrutura de convolução do híbrido
  model.add(layers.Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu',
    input_shape=(janela_tempo, num_features)
  ))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(0.2))
  
  model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(0.2))
  
  # "Achatamento" os dados para a camada Dense
  model.add(layers.Flatten())
  
  model.add(layers.Dense(units=25, activation='relu'))
  model.add(layers.Dense(units=1, activation='linear'))
  
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return model

def create_model_lstm(janela_tempo=60, num_features=5):
    model = models.Sequential()
    
    # A LSTM recebe direto o input bruto (60, 5)
    model.add(layers.LSTM(
      units=64,
      return_sequences=True, 
      input_shape=(janela_tempo, num_features)
    ))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.LSTM(units=32, return_sequences=False))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(units=25, activation='relu'))
    model.add(layers.Dense(units=1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_model_hybrid(janela_tempo=60, num_features=5):
  model = models.Sequential()
  
  # === PARTE 1: CNN (Extração de Padrões Espaciais/Morfológicos) ===
  # O Conv1D vai passar analisando os formatos dos candles
  model.add(layers.Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu',
    input_shape=(janela_tempo, num_features)
  ))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(0.2)) # Evita Overfitting
  
  model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(0.2))
  
  # === PARTE 2: LSTM (Análise da Memória Temporal/Tendência) ===
  # A LSTM recebe os padrões que a CNN limpou e analisa a sequência deles no tempo
  model.add(layers.LSTM(units=50, return_sequences=False))
  model.add(layers.Dropout(0.2))
  
  # === PARTE 3: Camadas Densas (Tomada de Decisão) ===
  model.add(layers.Dense(units=25, activation='relu'))
  
  # Saída: Previsão do próximo preço de fechamento
  model.add(layers.Dense(units=1, activation='linear')) 
  
  # Compilação do modelo
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  
  return model

def main():    
  # modelo_hibrido = create_model_hybrid(janela_tempo=60, num_features=5)
  # modelo_hibrido.summary()

  # modelo_cnn = create_model_cnn(janela_tempo=60, num_features=5)
  # modelo_cnn.summary()

  # modelo_lstm = create_model_lstm(janela_tempo=60, num_features=5)
  # modelo_lstm.summary()

  pipeline() # Executa o pipeline de extração de dados

if __name__ == "__main__":
  main()
