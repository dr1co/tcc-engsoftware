# TCC — Previsão de preços no mercado futuro brasileiro com Deep Learning

Trabalho de Conclusão de Curso (Engenharia de Software — UEPG) que constrói e avalia
modelos de Deep Learning (CNN, LSTM e um híbrido CNN+LSTM) para prever o próximo
preço de fechamento de contratos futuros brasileiros — mini dólar (WDO) e mini
índice (WIN) — a partir de candles de 1 hora (H1).

O documento oficial do trabalho está em [`TCC_Adriano_2026.pdf`](TCC_Adriano_2026.pdf).
As decisões de implementação não especificadas no artigo (ajuste do scaler, seleção
de features, configuração de treino, metodologia de backtesting etc.), junto com o
racional de cada uma, estão documentadas em `NOTAS.md` (não versionado — ver seção
"Documentação interna" abaixo).

## Estrutura do projeto

```
.
├── data/                       # CSVs brutos de candles (dados_{WDON,WINN}_H1.csv)
├── models/{WDON,WINN}/         # modelos treinados (.keras), scalers (.pkl), métricas (.json)
├── results/                    # tabelas comparativas (.csv) e curvas de equity do backtest
├── src/
│   ├── extract_data.py         # extração via MetaTrader 5 (mt5linux)
│   ├── data_formatting.py      # limpeza/sanitização dos dados brutos
│   ├── features.py             # indicadores técnicos (RSI, SMA, MACD)
│   ├── utils.py                # janelamento temporal (sliding windows)
│   ├── models.py               # arquiteturas CNN / LSTM / híbrida (Keras)
│   ├── train.py                # orquestração do treino
│   ├── evaluate.py             # métricas de erro (RMSE, MAE, acurácia direcional)
│   ├── backtest.py             # métricas de negócio (Sharpe, Drawdown, VaR)
│   ├── tune.py                 # busca de hiperparâmetros (Keras Tuner)
│   └── pipelines/              # scripts replicáveis, um por cenário experimental
├── main.py                     # ponto de entrada (treina os 6 modelos)
├── requirements.txt
├── CLAUDE.md                   # guia de contexto para o Claude Code (inglês)
└── NOTAS.md                    # racional detalhado de cada decisão (não versionado)
```

### Estágios centrais do pipeline

| # | Módulo | O que faz |
|---|---|---|
| 1 | `src/extract_data.py` | Extrai candles OHLCV do MetaTrader 5 e salva em `data/dados_{MERCADO}_H1.csv` |
| 2 | `src/data_formatting.py` | Ordena, remove duplicatas/candles inconsistentes, trata valores ausentes |
| 3 | `src/utils.py` | Converte a série em janelas deslizantes `(60 candles, N features)` |
| 4 | `src/models.py` | Constrói os modelos Keras (CNN, LSTM, híbrido) |
| 5 | `src/train.py` | Escalonamento (MinMaxScaler), split cronológico 70/15/15, treino com EarlyStopping |
| 6 | `src/evaluate.py` / `src/backtest.py` | Métricas de erro e de negócio sobre os modelos já treinados |

`src/tune.py` (busca de hiperparâmetros) e `src/features.py` (indicadores técnicos)
são etapas experimentais adicionais — ver `src/pipelines/` e `NOTAS.md` para o
contexto de cada uma.

### Gerações de modelos

Cada rodada de treino é identificada por um sufixo, e as gerações coexistem em
`models/{MERCADO}/` sem se sobrescrever:

| Sufixo | Features de entrada | Descrição |
|---|---|---|
| *(nenhum)* | preço em nível absoluto | Geração original, substituída — mantida só como comparação |
| `_log` | 5 (OHLC em log-retorno + volume) | Geração-base de todo o restante do experimento |
| `_tuned` | 5 (mesmas de `_log`) | Arquitetura/config de treino trocada pela melhor combinação encontrada via busca de hiperparâmetros |
| `_features` | 10 (5 base + RSI/SMA/MACD) | Indicadores técnicos adicionados como features extras |

### Cenários experimentais replicáveis

`src/pipelines/` contém um script por cenário testado, cada um com docstring
explicando motivação, fluxo, artefatos produzidos e o resultado de referência
obtido. Rodar com `python -m src.pipelines.NN_nome` (ver `src/pipelines/__init__.py`
para a ordem de dependência entre eles):

1. `01_baseline_log` — treino/avaliação/backtest da geração `_log`.
2. `02_confidence_filter` — filtro de confiança por magnitude do retorno previsto.
3. `03_ensemble_agreement` — filtro por concordância entre as 3 arquiteturas.
4. `04_hyperparameter_tuning` — busca de hiperparâmetros (gera a geração `_tuned`).
5. `05_technical_indicators` — indicadores técnicos (gera a geração `_features`).
6. `06_combined_final_check` — filtro de confiança combinado com `_tuned`/`_features`,
   com correção estatística para comparações múltiplas (Bonferroni).

**Resultado obtido:** nenhuma das cinco abordagens produziu uma vantagem direcional
estatisticamente confiável para WIN/WDO no horizonte de 1 hora testado — ver
`NOTAS.md` (decisões 17, 18, 25, 27, 28) para a análise completa.

## Ambiente e configuração

- **Python:** 3.13.15 (fixado em `.python-version`).
- **Ambiente virtual:** já existe em `.venv/`. Ativar antes de rodar qualquer comando:
  ```bash
  source .venv/bin/activate
  ```
- **Dependências:** declaradas em `requirements.txt` (UTF-8, `pip install -r requirements.txt`).
  Principais bibliotecas:
  - `tensorflow` / `keras` — construção e treino dos modelos.
  - `keras-tuner` — busca de hiperparâmetros.
  - `pandas` / `numpy` — manipulação de dados.
  - `pandas-ta` — cálculo de indicadores técnicos (RSI, SMA, MACD).
  - `scikit-learn` — `MinMaxScaler`.
  - `scipy` — teste-t de significância estatística no backtest.
  - `joblib` — serialização do scaler.
  - `rich` — barras de progresso durante a busca de hiperparâmetros.

  > **Nota:** instalar `pandas-ta` rebaixa o `numpy` (de 2.4.6 para 2.2.6, via sua
  > dependência `numba`) — comportamento esperado e já validado como compatível com
  > o restante do ambiente, não é um erro de instalação.

- **Extração de dados (opcional):** `src/extract_data.py` depende do pacote
  `mt5linux` (não incluído em `requirements.txt`, pois só é necessário para
  reextrair os dados brutos) e de uma ponte para um terminal MetaTrader 5 rodando
  em `localhost:18812` (tipicamente via Wine/Bottles). Os CSVs já extraídos estão
  em `data/`; só é necessário rodar essa etapa para atualizar o histórico de candles.

### Observação sobre WSL/Windows

Se este repositório for clonado em ambiente Windows, é preferível utilizá-lo sob
o sistema de arquivos do WSL. No entanto, se estiver executando
comandos a partir do lado Windows (ex.: PowerShell contra o caminho
`\\wsl.localhost\Ubuntu\...`), `source .venv/bin/activate` pode resolver para um
Python diferente do ambiente real do projeto. Para evitar isso, invoque os comandos
explicitamente via WSL:

```bash
wsl -e bash -c "cd ~/caminho/para/tcc-engsoftware && source .venv/bin/activate && python3 ..."
```

## Como rodar

Treinar os 6 modelos (3 arquiteturas × 2 mercados) da geração `_log`:

```bash
python main.py
```

Avaliar (RMSE, MAE, acurácia direcional):

```bash
python -m src.evaluate
```

Rodar o backtest (Sharpe Ratio, Maximum Drawdown, VaR):

```bash
python -m src.backtest
```

Rodar a busca de hiperparâmetros (demorado — horas em CPU):

```bash
python -m src.tune
```

Replicar um cenário experimental específico:

```bash
python -m src.pipelines.01_baseline_log
```

Não há suíte de testes, linter ou formatter configurados neste repositório.

## Documentação interna

- **`CLAUDE.md`** — guia de contexto em inglês para uso com o Claude Code, cobrindo
  a arquitetura do pipeline em maior nível de detalhe técnico.
- **`NOTAS.md`** — registro detalhado do racional por trás de cada decisão de
  implementação não especificada no artigo (ajuste do scaler, seleção de features,
  metodologia de backtesting, resultados de cada experimento etc.), escrito para
  reaproveitamento direto nas seções de Metodologia e Análise de Resultados do TCC.
  Não é versionado no git (uso pessoal/rascunho).
