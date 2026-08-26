"""
Pipelines de replicação dos cenários experimentais do TCC.

Cada módulo neste pacote corresponde a um cenário/etapa distinto do
experimento, documentado em detalhe em NOTAS.md, e pode ser executado de
forma independente (`python -m src.pipelines.NN_nome_do_cenario`) para
reproduzir exatamente os resultados obtidos naquela etapa. Os módulos
apenas orquestram funções já existentes em src/train.py, src/evaluate.py,
src/backtest.py e src/tune.py — nenhuma lógica de modelagem/backtesting é
duplicada aqui, só a sequência de chamadas que caracteriza cada cenário.

Ordem cronológica dos cenários (prefixo numérico nos nomes de arquivo):

  01_baseline_log       - Geracao "_log": features em log-retorno (5 colunas),
                          sem tuning, sem filtro de confianca. Baseline de
                          todo o restante do experimento (NOTAS.md decisoes 1-16).

  02_confidence_filter  - Filtro de confianca por magnitude de |log-retorno
                          previsto| aplicado sobre a geracao "_log" (NOTAS.md
                          decisao 17). Primeira tentativa de melhorar o Sharpe
                          sem retreinar nada.

  03_ensemble_agreement - Filtro de confianca por concordancia de direcao entre
                          as 3 arquiteturas (unanimidade 3/3) aplicado sobre a
                          geracao "_log" (NOTAS.md decisao 18). Segunda tentativa
                          de seletividade de trades, medida de confianca diferente.

  04_hyperparameter_tuning - Busca de hiperparametros (Keras Tuner RandomSearch)
                          por (mercado, arquitetura), gerando a geracao "_tuned"
                          (NOTAS.md decisoes 19-25). Ataca a capacidade/configuracao
                          do modelo em vez da regra de decisao sobre as previsoes.

  05_technical_indicators - Adiciona RSI/SMA/MACD como features extras (10 no
                          total), gerando a geracao "_features" (NOTAS.md
                          decisoes 26-27). Ataca a informacao de entrada em vez
                          da capacidade do modelo ou da regra de decisao.

  06_combined_final_check - Reaplica o filtro de confianca por magnitude (mesmo
                          metodo do cenario 02) sobre as geracoes "_tuned" e
                          "_features", e verifica os resultados agregados das 3
                          geracoes com correcao para comparacoes multiplas
                          (Bonferroni) antes de aceitar qualquer resultado
                          aparentemente positivo (NOTAS.md decisao 28) - o
                          cenario de fechamento da fase de experimentacao.

Pré-requisitos comuns a todos os cenários:
  - CSVs brutos em data/dados_{WDON,WINN}_H1.csv (ver src/extract_data.py).
  - Ambiente com as dependências de requirements.txt instaladas.
  - Os cenários 02, 03 e 06 esperam que os modelos da(s) geração(ões) que
    consomem já tenham sido treinados (rode 01/04/05 antes, conforme o caso).
"""
