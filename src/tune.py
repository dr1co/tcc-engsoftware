import os
import json

import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from src.utils import create_time_windows
from src.models import create_model_cnn, create_model_lstm, create_model_hybrid
from src.train import FEATURE_COLS, TIMEFRAME, load_prepared, split_chronological

MODEL_BUILDERS = {
  'cnn': create_model_cnn,
  'lstm': create_model_lstm,
  'hybrid': create_model_hybrid,
}

MAX_TRIALS = 18
MAX_EPOCHS = 200
PATIENCE_CHOICES = [8, 12, 16]
BATCH_SIZE_CHOICES = [16, 32, 64]


class MarketHyperModel(kt.HyperModel):
  """
  Envolve create_model_* (src/models.py) para o Keras Tuner: build() delega a
  construcao do modelo (arquitetura + otimizador) para a mesma funcao usada em
  produção, passando o objeto hp para que ela defina seu proprio espaco de busca
  (ver create_model_cnn/lstm/hybrid). fit() adiciona batch_size e patience do
  EarlyStopping como hiperparametros tunables tambem, já que RandomSearch só
  varia o que é declarado via hp em algum lugar do ciclo build/fit.
  """

  def __init__(self, architecture):
    self.architecture = architecture
    self.build_fn = MODEL_BUILDERS[architecture]

  def build(self, hp):
    return self.build_fn(timeframe=TIMEFRAME, num_features=len(FEATURE_COLS), hp=hp)

  def fit(self, hp, model, X_train, y_train, validation_data, **kwargs):
    batch_size = hp.Choice('batch_size', BATCH_SIZE_CHOICES)
    patience = hp.Choice('patience', PATIENCE_CHOICES)

    early_stopping = EarlyStopping(
      monitor='val_loss',
      patience=patience,
      restore_best_weights=True,
    )

    # o Keras Tuner injeta seus proprios callbacks (ex.: TensorBoard) via **kwargs;
    # 'callbacks' pode ja vir preenchido, entao acrescentamos o EarlyStopping em vez
    # de sobrescrever, senao model.fit recebe 'callbacks' duplicado (TypeError).
    kwargs['callbacks'] = kwargs.get('callbacks', []) + [early_stopping]
    kwargs.setdefault('verbose', 0)  # log por epoca silenciado - progresso mostrado via barra rich

    return model.fit(
      X_train, y_train,
      validation_data=validation_data,
      batch_size=batch_size,
      epochs=MAX_EPOCHS,
      **kwargs,
    )


class _ProgressRandomSearch(kt.RandomSearch):
  """
  RandomSearch com um hook em on_trial_end para avancar uma barra de progresso
  rich por trial concluido, ja que tuner.search() e uma chamada bloqueante que
  roda todos os trials internamente (sem retorno de progresso incremental por
  padrao). trial_progress_cb(trial) e chamado apos cada trial terminar.
  """

  def __init__(self, *args, trial_progress_cb=None, **kwargs):
    super().__init__(*args, **kwargs)
    self._trial_progress_cb = trial_progress_cb

  def on_trial_end(self, trial):
    super().on_trial_end(trial)
    if self._trial_progress_cb is not None:
      self._trial_progress_cb(trial)


def tune_one_model(market, architecture, max_trials=MAX_TRIALS, project_dir='tuning', progress=None):
  """
  Roda RandomSearch (Keras Tuner) para um par (mercado, arquitetura), otimizando
  val_loss (MSE) - mesma metrica que EarlyStopping ja monitora em train.py. Retorna
  os melhores hiperparametros encontrados (kt.HyperParameters) e o dict equivalente.

  progress (opcional, uma instancia de rich.progress.Progress) permite mostrar uma
  barra aninhada de progresso por trial dentro da barra geral de tune_all(); quando
  omitido, cria e gerencia sua propria barra local (uso direto/scripts).

  Nao salva o modelo final "vencedor" automaticamente - ver tune_all()/apply_best_hp,
  que retreinam com train_one_model usando os hiperparametros encontrados aqui e
  salvam como geracao "_tuned" (mantendo "_log" como baseline, ver NOTAS.md). O
  avanco da barra de nivel "par" (se houver) e responsabilidade de quem chama esta
  funcao (tune_all), pois so deve avancar apos apply_best_hp tambem terminar - esta
  funcao sozinha so cobre a busca, nao o retreino final.
  """
  print(f"\n=== Tuning {architecture.upper()} para {market} ({max_trials} trials) ===")

  df, _ = load_prepared(market)
  df_train, df_val, _ = split_chronological(df)

  scaler = MinMaxScaler(feature_range=(0, 1))
  train_scaled = scaler.fit_transform(df_train.values)
  val_scaled = scaler.transform(df_val.values)

  X_train, y_train = create_time_windows(train_scaled, timeframe=TIMEFRAME)
  X_val, y_val = create_time_windows(val_scaled, timeframe=TIMEFRAME)

  hypermodel = MarketHyperModel(architecture)

  def _own_progress():
    p = Progress(
      TextColumn("[progress.description]{task.description}"),
      BarColumn(),
      MofNCompleteColumn(),
      TimeElapsedColumn(),
    )
    return p

  own_progress = progress is None
  local_progress = _own_progress() if own_progress else progress
  trial_task = local_progress.add_task(f"  trials [{market}/{architecture}]", total=max_trials)

  if own_progress:
    local_progress.start()

  def _on_trial_end(trial):
    local_progress.advance(trial_task)

  try:
    tuner = _ProgressRandomSearch(
      hypermodel,
      objective='val_loss',
      max_trials=max_trials,
      executions_per_trial=1,
      directory=project_dir,
      project_name=f"{market}_{architecture}",
      overwrite=True,
      trial_progress_cb=_on_trial_end,
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val))
  finally:
    if own_progress:
      local_progress.stop()
    else:
      local_progress.remove_task(trial_task)

  best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
  print(f"Melhores hiperparametros para {architecture.upper()}/{market}: {best_hp.values}")

  return best_hp


def apply_best_hp(market, architecture, best_hp, suffix='_tuned'):
  """
  Retreina o modelo final com os melhores hiperparametros encontrados por
  tune_one_model, salvando como uma nova geracao de artefatos (padrao "_tuned"),
  e persiste os proprios hiperparametros em JSON para documentacao/reproducibilidade.
  Reaproveita train_one_model importando localmente para evitar import circular
  (train.py nao depende de tune.py).
  """
  from src.train import train_one_model

  hp_dict = dict(best_hp.values)
  batch_size = hp_dict.get('batch_size', 32)
  patience = hp_dict.get('patience', 12)

  out_dir = f"models/{market}"
  os.makedirs(out_dir, exist_ok=True)
  hp_path = f"{out_dir}/{architecture}_best_hp.json"
  with open(hp_path, 'w', encoding='utf-8') as f:
    json.dump(hp_dict, f, indent=2, ensure_ascii=False)
  print(f"Hiperparametros salvos em: {hp_path}")

  # train_one_model constroi o modelo via MODEL_BUILDERS[architecture](timeframe, num_features)
  # sem hp, ou seja, sem os valores tunados. Para aplicar best_hp no treino final,
  # sobrepomos temporariamente o builder daquela arquitetura por uma versao presa
  # (partial) que já injeta os hiperparametros encontrados como hp fixo.
  import src.train as train_module
  original_builder = train_module.MODEL_BUILDERS[architecture]

  def _fixed_builder(timeframe, num_features, _hp_dict=hp_dict):
    return MODEL_BUILDERS[architecture](
      timeframe=timeframe,
      num_features=num_features,
      hp=_FixedHP(_hp_dict),
    )

  train_module.MODEL_BUILDERS[architecture] = _fixed_builder
  try:
    return train_one_model(market, architecture, batch_size=batch_size, patience=patience, suffix=suffix)
  finally:
    train_module.MODEL_BUILDERS[architecture] = original_builder


class _FixedHP:
  """
  Substituto minimo para kt.HyperParameters que devolve valores ja escolhidos
  (best_hp.values) em vez de amostrar um novo valor - permite reusar
  create_model_cnn/lstm/hybrid tal como estao (elas so chamam hp.Choice/hp.Float)
  para reconstruir o modelo vencedor fora do laco de busca do Keras Tuner.
  """

  def __init__(self, values):
    self.values = values

  def Choice(self, name, options):
    return self.values[name]

  def Float(self, name, min_value, max_value, step=None, sampling=None):
    return self.values[name]


def tune_all(max_trials=MAX_TRIALS, suffix='_tuned'):
  """
  Roda tune_one_model + apply_best_hp para as 6 combinacoes (3 arquiteturas x 2
  mercados), salvando os modelos vencedores como geracao "_tuned" e os
  hiperparametros de cada um em models/{market}/{architecture}_best_hp.json.

  Mostra duas barras de progresso rich: uma para os 6 pares mercado/arquitetura
  e uma aninhada para os trials da busca em andamento (ver tune_one_model). Os
  logs por epoca do Keras ficam silenciados durante a busca (MarketHyperModel.fit
  usa verbose=0) para nao competir visualmente com as barras; o retreino final de
  cada par (apply_best_hp -> train_one_model) mantem verbose=2, ja que roda uma
  unica vez por par e nao esta dentro do laco de busca.
  """
  markets = ['WDON', 'WINN']
  architectures = ['cnn', 'lstm', 'hybrid']
  pairs = [(market, architecture) for market in markets for architecture in architectures]

  results = {}
  with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
  ) as progress:
    pair_task = progress.add_task("Tuning (pares mercado/arquitetura)", total=len(pairs))

    for market, architecture in pairs:
      best_hp = tune_one_model(market, architecture, max_trials=max_trials, progress=progress)
      apply_best_hp(market, architecture, best_hp, suffix=suffix)
      results[(market, architecture)] = dict(best_hp.values)
      progress.advance(pair_task)

  return results


if __name__ == "__main__":
  tune_all()
