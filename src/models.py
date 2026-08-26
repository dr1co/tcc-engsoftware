import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def create_model_cnn(timeframe=60, num_features=5, hp=None):
  filters_1 = hp.Choice('conv1_filters', [32, 64, 128, 256]) if hp else 64
  filters_2 = hp.Choice('conv2_filters', [16, 32, 64, 128]) if hp else 32
  kernel_size = hp.Choice('kernel_size', [2, 3, 4, 5, 7]) if hp else 3
  dense_units = hp.Choice('dense_units', [16, 25, 32, 64]) if hp else 25
  dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.05) if hp else 0.2
  learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log') if hp else 1e-3

  model = models.Sequential()

  # Mesma estrutura de convolução do híbrido
  model.add(layers.Conv1D(
    filters=filters_1,
    kernel_size=kernel_size,
    activation='relu',
    input_shape=(timeframe, num_features)
  ))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(dropout_rate))

  model.add(layers.Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(dropout_rate))

  # "Achatamento" os dados para a camada Dense
  model.add(layers.Flatten())

  model.add(layers.Dense(units=dense_units, activation='relu'))
  model.add(layers.Dense(units=1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
  return model

def create_model_lstm(timeframe=60, num_features=5, hp=None):
  units_1 = hp.Choice('lstm1_units', [32, 64, 96, 128]) if hp else 64
  units_2 = hp.Choice('lstm2_units', [16, 32, 64]) if hp else 32
  dense_units = hp.Choice('dense_units', [16, 25, 32, 64]) if hp else 25
  dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.05) if hp else 0.2
  learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log') if hp else 1e-3

  model = models.Sequential()

  # A LSTM recebe direto o input bruto (60, 5)
  model.add(layers.LSTM(
    units=units_1,
    return_sequences=True,
    input_shape=(timeframe, num_features)
  ))
  model.add(layers.Dropout(dropout_rate))

  model.add(layers.LSTM(units=units_2, return_sequences=False))
  model.add(layers.Dropout(dropout_rate))

  model.add(layers.Dense(units=dense_units, activation='relu'))
  model.add(layers.Dense(units=1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
  return model

def create_model_hybrid(timeframe=60, num_features=5, hp=None):
  filters_1 = hp.Choice('conv1_filters', [32, 64, 128, 256]) if hp else 64
  filters_2 = hp.Choice('conv2_filters', [16, 32, 64, 128]) if hp else 32
  kernel_size = hp.Choice('kernel_size', [2, 3, 4, 5, 7]) if hp else 3
  lstm_units = hp.Choice('lstm_units', [32, 50, 64, 96]) if hp else 50
  dense_units = hp.Choice('dense_units', [16, 25, 32, 64]) if hp else 25
  dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.05) if hp else 0.2
  learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log') if hp else 1e-3

  model = models.Sequential()

  # === PARTE 1: CNN (Extração de Padrões Espaciais/Morfológicos) ===
  # O Conv1D vai passar analisando os formatos dos candles
  model.add(layers.Conv1D(
    filters=filters_1,
    kernel_size=kernel_size,
    activation='relu',
    input_shape=(timeframe, num_features)
  ))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(dropout_rate)) # Evita Overfitting

  model.add(layers.Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Dropout(dropout_rate))

  # === PARTE 2: LSTM (Análise da Memória Temporal/Tendência) ===
  # A LSTM recebe os padrões que a CNN limpou e analisa a sequência deles no tempo
  model.add(layers.LSTM(units=lstm_units, return_sequences=False))
  model.add(layers.Dropout(dropout_rate))

  # === PARTE 3: Camadas Densas (Tomada de Decisão) ===
  model.add(layers.Dense(units=dense_units, activation='relu'))

  # Saída: Previsão do próximo preço de fechamento
  model.add(layers.Dense(units=1, activation='linear'))

  # Compilação do modelo
  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

  return model
