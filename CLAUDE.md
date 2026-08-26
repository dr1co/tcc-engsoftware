> **Keep this file up to date!** Whenever a new feature/pipeline is implemented, a rule is changed. Keep this document up to date. Treat a stale document as a critical bug, as a stale document is worse than none at all.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a TCC (undergraduate thesis, "Trabalho de Conclusão de Curso") in Software Engineering at UEPG. It builds deep learning models (CNN, LSTM, and a hybrid CNN+LSTM) to predict the next close price of Brazilian futures contracts — mini dollar (WDO) and mini índice (WIN) — from hourly (H1) candle data. Code comments, print output, and docstrings are in Portuguese; keep new comments/output consistent with that unless told otherwise.

See [README.md](README.md) (Portuguese) for project structure, environment setup, and dependencies. See `NOTAS.md` for the full rationale behind every pipeline decision not specified in the thesis — read it before changing training/evaluation/backtest behavior.

## Environment gotchas

- **WSL/Windows path gotcha:** this repo's real `.venv` can live on the WSL filesystem (pyenv path under `/home/...`) if running on Windows. Shell tools running from the Windows side against the `\\wsl.localhost\Ubuntu\...` UNC path can silently resolve `source .venv/bin/activate` to a different or broken Python. To reliably reach the real venv, invoke through WSL explicitly: `wsl -e bash -c "cd ~/path/to/this/repository && source .venv/bin/activate && python3 ..."`. If a just-installed package can't be imported, this mismatch is the first thing to check.
- Data extraction (`src/extract_data.py`) depends on `mt5linux`, a bridge to a MetaTrader 5 terminal under Wine/Bottles at `localhost:18812` — not runnable in a typical sandbox. Don't attempt it unless that bridge is confirmed available.
- Installing `pandas-ta` pins `numpy` down to 2.2.6 (via its `numba` dependency) — verified compatible with the rest of the stack, but expect it if a future `pip install` seems to "downgrade" numpy.

## Architecture

Nine pipeline stages, each isolated under `src/`, in dependency order:

1. **`src/extract_data.py`** — pulls OHLCV from MetaTrader 5 into `data/dados_{MARKET}_H1.csv`. Run once, out-of-band.
2. **`src/data_formatting.py`** — `sanitize_df()`: cleans a raw CSV (dedup, NaN fill, drop inconsistent candles), adds informational `log_return`/`candle_range` columns.
3. **`src/utils.py`** — `create_time_windows()`: turns a 2D feature array into 3D sliding windows for sequence models.
4. **`src/models.py`** — `create_model_cnn/lstm/hybrid(timeframe, num_features, hp=None)`: three Keras builders. `hp=None` uses the original hardcoded defaults (Tables 1–3 in the thesis); pass a Keras Tuner `hp` object to make any architecture parameter searchable (used by stage 8).
5. **`src/train.py`** — orchestrates sanitize → feature prep → chronological 70/15/15 split → `MinMaxScaler` (fit on train only) → windowing → train with `EarlyStopping` → save. `train_one_model(market, architecture, suffix, feature_set)` / `train_all()`. `feature_set='base'` (5 cols, OHLC log-return + volume) or `'extended'` (+5 technical indicators, stage 9). `suffix` distinguishes model generations on disk — see **Model generations** below. `load_prepared()`, `prepare_features()`, `reconstruct_price()`, `FEATURE_SETS` are the shared building blocks `evaluate.py`/`backtest.py`/`tune.py` also import.
6. **`src/evaluate.py`** — `evaluate_one_model()` / `evaluate_all()`: RMSE/MAE/directional accuracy in real price units, decoupled from training (reloads saved `.keras`+`scaler.pkl`).
7. **`src/backtest.py`** — simulates trading (`single_bar`, `multi_bar` strategies) and computes Sharpe/Max Drawdown/VaR 95%/t-test p-value. `backtest_one_model()` / `backtest_all()`; `backtest_sweep()` adds a confidence-magnitude filter grid; `backtest_ensemble_all()` filters by 3-architecture unanimity.
8. **`src/tune.py`** — Keras Tuner `RandomSearch` per (market, architecture), optimizing `val_loss`. `tune_one_model()` / `tune_all()`; `apply_best_hp()` retrains the winning config and saves it as the `_tuned` generation.
9. **`src/features.py`** — `add_technical_indicators()`: RSI/SMA/MACD computed via `pandas_ta`, feeding `feature_set='extended'`.

**`src/pipelines/`** holds one runnable script per experimental scenario (`01_baseline_log.py` … `06_combined_final_check.py`), each a thin orchestrator with a docstring covering motivation, flow, artifacts, and reference results. Run via `python -m src.pipelines.NN_name`; see `src/pipelines/__init__.py` for the dependency order between them. This is the fastest way to replicate any part of the experiment — prefer it over re-deriving a call sequence from the modules above.

### Model generations

Every training run is tagged with a `suffix`, and generations coexist on disk without overwriting each other:

| Suffix | Feature set | What it is |
|---|---|---|
| *(none)* | raw price, Min-Max scaled | Original baseline — superseded, kept only for comparison (see `NOTAS.md` decision 12 for why it was replaced) |
| `_log` | `base` | OHLC in log-return + raw volume — the working baseline for everything after |
| `_tuned` | `base` | `_log` architecture/training config replaced by Keras Tuner's winning search result |
| `_features` | `extended` | `_log` + RSI/SMA/MACD (10 input features instead of 5) |

`feature_set` passed to `train`/`evaluate`/`backtest` must match whatever the target `suffix` was actually trained with, or array shapes silently mismatch.

### Result: no statistically defensible directional edge found

Five independent angles — confidence filtering by prediction magnitude, by ensemble agreement, hyperparameter tuning, technical indicator features, and their combination (checked with Bonferroni correction for the ~216 total statistical tests run) — were tried against WIN/WDO at a 1-hour horizon. None produced a Sharpe ratio distinguishable from noise. Full analysis, numbers, and thesis-write-up guidance: `NOTAS.md` decisions 17, 18, 25, 27, 28.

## Common commands

```bash
python main.py                # trains all 6 models (_log generation) — see src/train.py::train_all
python -m src.evaluate         # RMSE/MAE/directional accuracy for the _log generation
python -m src.backtest         # Sharpe/MaxDrawdown/VaR backtest, 12 combinations
python -m src.tune             # hyperparameter search + retrain, ~108 runs, hours-long
python -m src.pipelines.01_baseline_log   # ...or replicate any scenario individually, see src/pipelines/
```

There is currently no test suite, linter, or formatter configured in this repo.

## Thesis document

[TCC_Adriano_2026.pdf](TCC_Adriano_2026.pdf) is the thesis write-up itself — the source of truth for what's been written up versus what still only exists as code/experiments. Do not assume anything; always ask the user how/what is to be implemented. Grill the user whenever necessary with mattpocock's skill `grill-me` or mattpocock's skill `grilling`.
