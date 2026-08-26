# Pipeline de treino/teste — decisões de implementação

Este documento registra as decisões tomadas para a primeira versão funcional do
pipeline de treino e teste dos 6 modelos (3 arquiteturas × 2 mercados: WDO e WIN).
Ele existe para servir de referência ao escrever as seções de Metodologia e
Análises do TCC, cobrindo pontos que o artigo (`TCC_Adriano_2026.pdf`) deixa em
aberto ou que só foram decididos durante a implementação.

## Contexto: o que já estava decidido no artigo

Estas decisões já constam na Seção 3 do artigo e não são novidade — estão aqui
só para referência rápida:

- Normalização Min-Max Scaler, aplicada de forma independente por ativo.
- Split cronológico (sem shuffle): 70% treino, 15% validação, 15% teste.
- Métricas de erro: RMSE, MAE e acurácia direcional. Métricas de negócio
  (Sharpe Ratio, Maximum Drawdown, VaR simplificado) via backtesting.
- 6 modelos = 3 arquiteturas (CNN, LSTM, híbrido CNN-LSTM) × 2 mercados
  (WIN, WDO), treinados de forma independente por mercado.

## Decisões tomadas nesta etapa (não especificadas no artigo)

### 1. Ajuste do scaler: apenas no conjunto de treino

O `MinMaxScaler` é ajustado (`fit`) somente com os dados de treino de cada
mercado, e então usado para transformar (`transform`) treino, validação e
teste.

**Por quê:** ajustar o scaler no dataset inteiro antes do split vaza
informação do futuro (range de preços de validação/teste) para o treino —
um data leak clássico em séries temporais. Ajustar só no treino é a prática
padrão e evita esse viés, mesmo que valores de validação/teste possam cair
levemente fora de [0, 1] quando o mercado se move além da faixa observada no
treino (comportamento esperado, não é erro).

**Para o TCC:** vale explicitar esse cuidado na seção de metodologia (3.2)
como justificativa metodológica contra data leakage — é um ponto que
avaliadores de trabalhos com séries temporais costumam questionar.

### 2. Conjunto de features: 5 features (OHLC + volume), sem `log_return`/`candle_range`

Os modelos usam exatamente as 5 variáveis já documentadas nas Tabelas 1–3 do
artigo: `open`, `high`, `low`, `close`, `volume` — tensor de entrada
`(60, 5)`. As features derivadas `log_return` e `candle_range`, já calculadas
em `src/data_formatting.py`, **não** entram no vetor de entrada dos modelos
nesta etapa.

**Por quê:** o artigo já documenta e sustenta (Tabelas 1–3, Seção 4) a
contagem de parâmetros treináveis dos 3 modelos com base em um tensor de
entrada `(60, 5)`. Usar 7 features mudaria essas tabelas e exigiria
retrabalho do texto já escrito.

**Para o TCC:** se decidir incorporar `log_return`/`candle_range` no futuro
(ex.: como experimento de ablação ou versão 2 do pipeline), documente como
uma variação explícita e regenere as Tabelas 1–3 com a nova contagem de
parâmetros — não substitua silenciosamente o que já foi escrito.

### 3. Coluna de volume: `real_volume`

Dentre as duas colunas de volume disponíveis nos CSVs (`tick_volume` e
`real_volume`), os modelos usam `real_volume`.

**Por quê:** o artigo (Seção 3.2) menciona "volume financeiro", que mapeia
mais diretamente para `real_volume` (volume financeiro efetivamente
negociado) do que para `tick_volume` (contagem de mudanças de preço, uma
métrica proxy específica do MT5). Verificado nos CSVs atuais
(`data/dados_WDON_H1.csv`, `data/dados_WINN_H1.csv`): `real_volume` está
100% populado e não-zero em ambos os mercados, então não há problema de
qualidade de dado que justificasse usar `tick_volume` como alternativa.

**Para o TCC:** ao descrever as variáveis de entrada na Seção 3.2, especifique
explicitamente "volume financeiro (`real_volume`)" para eliminar a ambiguidade
que o texto atual tem entre as duas colunas de volume do MetaTrader 5.

### 4. Treinamento: EarlyStopping + orçamento generoso de épocas

Cada um dos 6 modelos é treinado com um número máximo de épocas alto (ex.:
200), `EarlyStopping` monitorando `val_loss` (`patience` de 10–15 épocas,
`restore_best_weights=True`) e `batch_size=32`.

**Por quê:** CNN, LSTM e o híbrido convergem em ritmos diferentes; escolher um
número fixo de épocas por arquitetura seria arbitrário e não está
especificado no artigo. `EarlyStopping` deixa cada modelo parar no seu ponto
natural, reduzindo tanto overfitting (treino longo demais) quanto
underfitting (treino curto demais) sem exigir tuning manual por modelo.

**Para o TCC:** ao reportar hiperparâmetros de treino (Seção 3.4 ou Análises),
documente o `patience` e `max epochs` usados, e quantas épocas cada um dos 6
modelos efetivamente rodou até o early stop — é um dado relevante para a
discussão de convergência por arquitetura/mercado.

### 5. Organização dos artefatos treinados

Modelos, scalers e métricas são salvos em `models/`, organizados por mercado:

```
models/
  WDON/
    cnn.keras
    lstm.keras
    hybrid.keras
    scaler.pkl
  WINN/
    cnn.keras
    lstm.keras
    hybrid.keras
    scaler.pkl
```

**Por quê:** o scaler é ajustado por ativo (não por arquitetura), então faz
sentido morar junto dos 3 modelos daquele mercado em vez de repetido ou solto
na raiz. A estrutura aninhada por mercado deixa claro, ao navegar o
diretório, que os 3 modelos de um mesmo mercado compartilham o mesmo scaler
e os mesmos dados de origem.

### 6. Organização do código: `src/train.py` como orquestrador

Foi criado `src/train.py`, com uma função reutilizável
(ex.: `train_one_model(market, architecture)`) que executa a sequência
sanitização → escalonamento → janelamento → construção do modelo → fit →
avaliação → salvamento. `main.py` chama essa função em loop para as
2×3 combinações de mercado/arquitetura.

**Por quê:** mantém a mesma separação em estágios já usada no restante do
projeto (`extract_data.py` → `data_formatting.py` → `utils.py` →
`models.py`), em vez de concentrar lógica de orquestração em `main.py`.

### 7. Escopo desta etapa: apenas treinar e salvar os modelos

Esta primeira versão do pipeline cobre **somente** o treinamento e
salvamento dos 6 modelos. O cálculo de métricas (RMSE, MAE, acurácia
direcional) e a simulação de backtesting (Sharpe Ratio, Maximum Drawdown,
VaR simplificado) ficam para uma etapa seguinte, separada.

**Por quê:** confirmar que o pipeline de treino roda de ponta a ponta para
os 6 modelos é um marco por si só. Acoplar avaliação e backtesting ao mesmo
passo aumentaria a superfície de mudança antes de validar que o treinamento
básico funciona.

**Para o TCC:** isso é consistente com a Seção 4 (Análises preliminares) do
artigo, que já descreve a fase atual como validação estrutural antes da
integração com dados reais — o texto atual não precisa de ajuste por causa
dessa decisão, mas a próxima seção de Análises deverá descrever a etapa de
avaliação/backtesting como um passo posterior e distinto do treinamento.

## Etapa de avaliação (métricas de erro)

Com os 6 modelos treinados e salvos, esta etapa adiciona `src/evaluate.py`
para calcular as métricas de erro (RMSE, MAE, acurácia direcional) descritas
na Seção 3.3 do artigo. Sharpe Ratio, Maximum Drawdown e VaR (que exigem uma
camada de simulação de Backtesting) continuam fora de escopo — ver decisão 7
acima, ainda válida.

### 8. Métricas reportadas em unidade real de preço, não em escala [0, 1]

RMSE e MAE são calculados após inverter a escala das previsões e dos valores
reais de volta para preço (R$ para WDO, pontos para WIN), usando o scaler
salvo no treino.

**Por quê:** métricas calculadas em espaço escalado [0, 1] não são
interpretáveis por si só (ex.: "RMSE de 0.02" não diz nada sobre o erro em
reais) e não são comparáveis entre WIN e WDO, que têm faixas de preço muito
diferentes. Convertendo para unidade real, dá para comparar diretamente os
dois mercados e comunicar o erro em termos que fazem sentido para a análise
de negócio (Seção 3.5).

**Detalhe de implementação:** como o `MinMaxScaler` foi ajustado nas 5
colunas juntas (OHLC + volume), `scaler.inverse_transform()` exige um array
com as 5 colunas. Para inverter apenas a coluna de fechamento, monta-se um
array `(n, 5)` de zeros, coloca-se o vetor de interesse (previsão, valor real
ou último fechamento da janela) na posição da coluna `close` (índice 3), e
extrai-se essa mesma coluna depois do `inverse_transform` — as demais colunas
são preenchidas com zero e não influenciam o resultado, pois o
`MinMaxScaler` escala cada coluna de forma independente.

**Para o TCC:** ao reportar as métricas de erro (Seção 3.3/4), especifique
que RMSE e MAE estão em unidade real de preço (R$ para WDO, pontos de índice
para WIN), não em escala normalizada — evita ambiguidade na leitura dos
números pelos avaliadores.

### 9. Definição de acurácia direcional: relativa ao último fechamento conhecido da janela

A acurácia direcional compara o sinal de `(previsão - último_fechamento)`
com o sinal de `(valor_real - último_fechamento)`, onde `último_fechamento`
é o preço de fechamento real (já revertido de escala) do último candle da
janela de entrada usada para aquela previsão.

**Por quê:** essa é a pergunta que importa na prática — "dado o que eu sei
agora (o último fechamento), o modelo acertou se o preço vai subir ou
descer?" — e reflete como a previsão seria usada em uma decisão real de
operação. Comparar previsões consecutivas entre si (sem ancorar no último
preço real conhecido) não tem essa correspondência direta com uso prático.

**Para o TCC:** ao descrever a métrica de acurácia direcional (Seção 3.3),
defina explicitamente essa referência (último fechamento conhecido da janela
de entrada) — o artigo atual menciona a métrica mas não formaliza em relação
a quê a direção é medida.

### 10. Avaliação como etapa separada do treino: `src/evaluate.py`

A avaliação não foi acoplada ao final de `train_one_model()`. Em vez disso,
`src/evaluate.py` carrega os artefatos já salvos em `models/{market}/`
(`.keras` + `scaler.pkl`), reconstrói o conjunto de teste a partir do zero
(mesma sanitização + split usados no treino) e calcula as métricas.

**Por quê:** desacopla avaliação de treino — é possível reavaliar um modelo
já treinado sem precisar retreiná-lo (por exemplo, se a lógica de métricas
mudar). O custo extra de reprocessar `sanitize_df` e refazer o split é
desprezível no tamanho atual dos dados (~12 mil candles por mercado).

**Para o TCC:** não afeta diretamente o texto da metodologia, mas reforça
que o pipeline de teste é reprodutível a partir apenas dos artefatos salvos
e dos CSVs brutos — relevante para a seção de reprodutibilidade, se houver.

### 11. Armazenamento dos resultados: JSON por modelo + tabela comparativa

Cada avaliação salva `models/{market}/{architecture}_metrics.json` com as
3 métricas. Ao final, `evaluate_all()` consolida os 6 resultados em
`results/comparison.csv` (colunas: `market, architecture, rmse, mae,
directional_accuracy`).

**Por quê:** o JSON por modelo mantém o resultado junto do artefato que o
gerou (útil para rastreabilidade). A tabela CSV combinada é o formato mais
direto para colar na Seção 3.5 do artigo, que descreve explicitamente uma
comparação entre WIN$ e WDO$ nas métricas de erro e acurácia.

**Para o TCC:** `results/comparison.csv` é a fonte primária para a tabela
comparativa da Seção 3.5 — os números já vêm em unidade real de preço
(decisão 8), então dá para citar diretamente no texto sem conversão manual.

### Resultado da primeira execução (informativo, não normativo)

Rodando `python -m src.evaluate` sobre os 6 modelos já treinados (ver
`results/comparison.csv` para os valores exatos): RMSE/MAE em unidade real
de preço (dezenas de R$ para WDO, milhares de pontos para WIN — compatível
com a escala de preço de cada ativo) e acurácia direcional próxima de 0.50
em todos os 6 modelos. Isso é esperado para uma primeira passada sem nenhum
tuning de hiperparâmetros — não é uma conclusão do trabalho, apenas confirma
que o pipeline de avaliação está funcionalmente correto. Acurácia direcional
próxima do acaso é um resultado a ser discutido/melhorado nas próximas
etapas (ex.: tuning, features adicionais, ou é o próprio ponto de partida
para a discussão da Seção 3.5).

**Nota (superada pela decisão 12 abaixo):** essa primeira leva de modelos foi
treinada sobre preço em nível absoluto (não log-retorno). Um problema
estrutural foi descoberto nela durante a etapa de backtesting — ver decisão
12. Os artefatos dessa geração (`models/{market}/{architecture}.keras`,
`scaler.pkl`, `{architecture}_metrics.json`, `results/comparison.csv`)
foram **mantidos no repositório** como registro de comparação/documentação,
não foram sobrescritos pela nova geração.

## Etapa de backtesting (métricas de negócio: Sharpe, Maximum Drawdown, VaR)

Com as métricas de erro validadas, o próximo passo natural — antes de partir
para tuning de hiperparâmetros — é a camada de backtesting descrita na Seção
3.3–3.4 do artigo (Sharpe Ratio, Maximum Drawdown, VaR simplificado), já que
é ela quem de fato diz se o sinal do modelo tem alguma utilidade prática, e é
o critério mais relevante para julgar se um tuning subsequente ajudou.

Durante a implementação dessa etapa, um problema estrutural sério foi
descoberto na primeira geração de modelos (a documentada nas decisões 1–11
acima) — ver decisão 12. Isso motivou uma mudança de metodologia (features
em log-retorno) e uma nova geração de modelos, mantendo a anterior como
registro comparativo.

### 12. Descoberta: extrapolação do Min-Max Scaler além do intervalo de treino (WIN)

**O problema encontrado:** ao rodar o backtest sobre os 6 modelos da primeira
geração, os 3 modelos de WIN (CNN, LSTM, híbrido) produziram resultados
**idênticos** entre si — mesmo Sharpe, mesmo Maximum Drawdown, mesmo VaR para
as 3 arquiteturas. Investigando a causa: os 3 modelos de WIN previam um
fechamento **abaixo do último fechamento conhecido em 100% das janelas de
teste** (0% de previsões de alta), um comportamento degenerado de "sempre
vender" independente da arquitetura.

**Diagnóstico:** o `MinMaxScaler` é ajustado (decisão 1) apenas com o
intervalo de preços observado no conjunto de treino. O WIN, diferente do WDO,
esteve em forte tendência de alta ao longo do período coletado — o preço de
fechamento no conjunto de teste (156.650 a 202.955 pontos) ficou **inteiramente
acima** do máximo observado no treino (96.300 a 139.165 pontos). Ao escalar o
conjunto de teste com o scaler ajustado no treino, os valores de fechamento
caíram entre 1,41 e 2,49 — muito fora do intervalo [0, 1] em que o modelo foi
efetivamente treinado. Redes neurais com camadas densas/convolucionais/LSTM
não extrapolam linearmente fora do domínio de treino; o resultado observado
foi um viés sistemático de subestimação (correlação de 0,976 entre previsão e
último fechamento real, porém com um offset negativo praticamente constante),
que em uma série de tendência de alta se traduz em "previsão sempre abaixo do
preço atual" em toda janela — daí a acurácia direcional de exatamente ~0,50
(a taxa-base de barras de queda no teste, não uma capacidade real do modelo)
e os resultados idênticos entre arquiteturas: as 3 redes convergiram para o
mesmo comportamento degenerado por causa do mesmo problema de escala, não por
coincidência de aprendizado.

Verificação: o WDO não apresentou esse problema nessa mesma divisão
cronológica porque, por coincidência do ponto de corte, o intervalo de preços
do teste (R$ 4.912,50 a R$ 5.598,50) ficou contido no intervalo do treino
(R$ 4.626,00 a R$ 6.297,50) — os valores escalados do teste ficaram entre
0,17 e 0,58, dentro de [0, 1]. Ou seja, **não é uma propriedade do WIN
enquanto mercado** (maior imprevisibilidade, etc.) — é uma falha estrutural
do pipeline (Min-Max Scaler sobre preço em nível absoluto) que só não se
manifestou no WDO por sorte da divisão temporal usada.

**A correção — features em log-retorno:** em vez de escalar `open`, `high`,
`low`, `close` em nível absoluto, cada uma passou a ser transformada em
log-retorno em relação ao seu próprio valor anterior:
`log(X_t / X_{t-1})`, implementado em `src/train.py::prepare_features`.
`real_volume` permanece em nível bruto (ainda escalado pelo Min-Max, sem o
mesmo problema de deriva). A ordem e a quantidade de colunas (5) foram
mantidas, preservando o formato de entrada `(60, 5)` documentado nas Tabelas
1–3 do artigo — não é necessário refazer essas tabelas.

**Por quê resolve o problema:** log-retorno é estacionário — sua distribuição
não se desloca com o nível absoluto do preço. Um retorno de +0,5% "parece
igual" seja o índice a 120.000 ou a 200.000 pontos. Isso significa que o
intervalo de valores do conjunto de teste permanece comparável ao do treino
independente de quanto o preço subiu desde então, eliminando a extrapolação
por construção — não é uma correção pontual para o WIN, é uma correção
estrutural que se aplica a qualquer mercado/período futuro em tendência forte.
Verificado: com log-retorno, o intervalo escalado do fechamento no teste do
WIN passou de [1,41, 2,49] para [0,07, 0,88] — dentro de [0, 1].

**Reconstrução de preço:** como o modelo agora prevê um log-retorno (não mais
um preço absoluto), a previsão de preço usada em RMSE/MAE/acurácia
direcional/backtest é reconstruída como
`preço_previsto = último_fechamento_real * exp(log_retorno_previsto)`
(`src/train.py::reconstruct_price`) — inversão direta da definição de
log-retorno. Essa reconstrução também não depende do nível absoluto do preço,
então continua correta mesmo que o mercado siga subindo além de qualquer
intervalo visto até agora.

**Nomenclatura dos artefatos:** a nova geração de modelos é salva com o
sufixo `_log` (`models/{market}/{architecture}_log.keras`,
`scaler_log.pkl`, `{architecture}_log_metrics.json`,
`results/comparison_log.csv`, `results/backtest_comparison_log.csv`), para
coexistir com os artefatos da geração anterior (sem sufixo) sem sobrescrevê-los.
`train_all()`, `evaluate_all()` e `backtest_all()` usam `_log` como padrão.

**Para o TCC:** esta é uma descoberta genuína e citável para a Seção 3.5
(Análise dos Resultados) — não é apenas um detalhe de implementação. Vale
documentar como um achado metodológico: (1) o problema (modelos de WIN
convergindo para um previsor degenerado por extrapolação de escala), (2) o
diagnóstico (intervalo de teste fora do intervalo de treino, verificável
numericamente), (3) a correção (mudança para features estacionárias via
log-retorno) e (4) a justificativa teórica (estacionariedade evita deriva de
escala independente do horizonte de tempo). Isso também justifica, a
posteriori, por que o artigo já cita "estacionariedade" como motivação para
`log_return` em `data_formatting.py` (Seção 3.2) — a motivação inicial era
teórica/geral, e esta etapa forneceu evidência empírica concreta de por que
ela importa nesse pipeline especificamente.

### 13. Regra de entrada: sinal direcional

A posição (comprada/vendida/nenhuma) é definida pelo sinal de
`(preço_previsto - último_fechamento_real)`: `sign > 0` → compra, `sign < 0`
→ venda, `sign == 0` → sem operação naquela janela.

**Por quê:** é a mesma referência já usada na acurácia direcional (decisão
9), garantindo consistência entre a métrica de erro e a regra de negociação
do backtest. É também a abordagem mais simples e mais comum na literatura de
CNN-LSTM citada no artigo, servindo como linha de base antes de qualquer
refinamento (ex.: filtro por magnitude mínima do movimento previsto).

### 14. Duas estratégias de simulação: automatizada (1 barra) e humana (N barras com stop/alvo)

Foram implementadas duas estratégias de backtest, ambas em `src/backtest.py`:

- **`single_bar` (automatizada):** entra na direção do sinal, mantém a
  posição por exatamente 1 barra (1 hora) e sai no fechamento da barra
  seguinte. Rebalanceamento a cada hora, sem regra de saída antecipada.
- **`multi_bar` (humana):** entra na direção do sinal, mantém por até
  `HOLD_BARS = 4` barras (4 horas), com stop-loss e take-profit fixados a
  uma distância igual ao `candle_range` médio do ativo no conjunto de
  **treino** (`(high - low)` médio) a partir do preço de entrada. Se o preço
  intrabarra (`high`/`low` de cada barra dentro da janela de holding) tocar o
  stop ou o alvo antes das 4 horas, a posição é encerrada naquele nível; caso
  contrário, sai no fechamento da 4ª barra.

**Por quê duas estratégias:** a automatizada é o caso mais simples de
mecanizar e serve de linha de base "pura" do sinal do modelo. A estratégia de
N barras com stop/alvo aproxima como um trader discricionário realmente
gerenciaria a posição (definir um objetivo e um limite de perda e não ficar
reagindo a cada novo candle), sendo mais realista para a discussão de
viabilidade prática do artigo (Seção 3.3, "viabilidade do modelo em cenários
reais em operações financeiras").

**Por quê o stop/alvo é baseado no `candle_range` médio do treino (não do
teste):** usar uma estatística do conjunto de teste para definir uma regra de
negociação aplicada sobre o próprio teste seria vazamento de informação — o
mesmo cuidado da decisão 1, aplicado aqui à calibração da estratégia. Usar o
`candle_range` médio do treino também faz a distância de stop/alvo escalar
naturalmente por ativo (pontos para WIN, R$ para WDO) sem precisar de
constantes separadas hardcoded por mercado.

**Por quê `HOLD_BARS = 4`:** aproximadamente meia sessão de pregão do WIN/WDO
na B3 (sessão de ~9h/dia), tempo suficiente para uma tendência real se
desenvolver além do ruído de curto prazo, mas ainda dentro do mesmo pregão —
evita expor o backtest a risco de gap overnight, que não está sendo modelado.

**Para o TCC:** a Seção 3.5 do artigo já menciona a construção de curvas de
capital (Equity Curves); os dados para esses gráficos estão salvos em
`results/equity_curves/{market}_{architecture}_{strategy}_log.csv`
(retorno por trade e P&L acumulado) — usar diretamente para gerar as figuras.

### 15. Custo de transação fixo por operação

Cada trade (independente da estratégia) deduz um custo fixo de round-trip do
seu P&L, definido por mercado em `TRANSACTION_COST` (`src/backtest.py`):
R$ 2,00 para WDO, 40 pontos para WIN.

**Por quê incluir custo:** sem custo de transação, uma estratégia rebalanceada
a cada hora com acurácia direcional próxima de 0,50 pode aparentar lucratividade
apenas por ruído estatístico — um resultado que não resistiria a uma banca
examinadora, já que não reflete a realidade de operar na B3 (spread,
corretagem, taxas de bolsa). Incluir um custo mínimo, mesmo que aproximado,
torna o Sharpe/Drawdown/VaR reportados mais defensáveis.

**Por quê esses valores especificamente:** são uma aproximação de ordem de
grandeza para corretagem + taxas de um contrato mini (WDO/WIN) na B3,
expressos na mesma unidade de preço de cada ativo (R$ para WDO, pontos-índice
para WIN) para manter o dedutor na mesma escala do P&L por trade. **Não** é um
valor de corretagem real de nenhuma corretora específica — é uma constante de
engenharia para evitar que o backtest reporte lucro artificial de ruído puro.

**Para o TCC:** ao reportar os resultados de backtest (Seção 3.5), declare
explicitamente que os valores de custo são uma aproximação, não uma tarifa
real de mercado — evita que um avaliador questione a fonte exata do número.
Se houver dados reais de corretagem/spread do book do WIN/WDO disponíveis,
vale substituir essa constante por um valor mais defensável.

### 16. Position sizing fixo, VaR histórico a 95%, Sharpe anualizado por horas de pregão B3

- **Position sizing:** todo trade opera o mesmo tamanho fixo (1 contrato
  nocional). Mantém o backtest focado em avaliar o sinal preditivo do modelo
  isoladamente, sem misturar com uma estratégia de dimensionamento de risco
  separada (ex.: sizing por volatilidade) — refinamento possível para uma
  etapa futura, não para esta primeira passada.
- **VaR:** histórico (percentil empírico da distribuição de retornos por
  trade), a 95% de confiança — convenção mais comum tanto na literatura
  acadêmica quanto no uso de mercado (ex.: estilo Basel), citado no artigo
  como "VaR simplificado".
- **Sharpe Ratio:** calculado por trade (`média / desvio-padrão` dos
  retornos) e anualizado multiplicando por
  `sqrt(horas de pregão por dia × dias de pregão por ano)` =
  `sqrt(9 × 250) ≈ sqrt(2250)`, usando a duração real do pregão do WIN/WDO na
  B3 (~9h/dia) em vez de uma convenção genérica de 24h ou de mercados
  americanos — mantém o fator de anualização consistente com o tipo de ativo
  do estudo.

**Para o TCC:** declare esses 3 parâmetros (sizing fixo, VaR 95% histórico,
fator de anualização `sqrt(2250)`) explicitamente na Seção 3.3/3.4 do artigo
ao descrever a metodologia de backtesting — nenhum deles está especificado no
texto atual, e todos afetam diretamente os números reportados na Seção 3.5.

## Etapa de refinamento da estratégia: filtro de confiança

Com o backtest baseline rodando (decisões 12–16), a observação de que os 6
modelos operam em praticamente todas as janelas do teste (1753 trades para
WDO, 1759 para WIN) motivou uma pergunta: isso é realista? Um trader real não
opera a cada hora — ele filtra por convicção. Isso levou a testar se
restringir os trades às previsões de maior "confiança" do próprio modelo
melhora o resultado do backtest.

### 17. Filtro de confiança por percentil de `|log-retorno previsto|`, testado com teste-t

**A ideia:** para cada modelo, `|predicted_log_return|` (a magnitude do
movimento previsto, antes da reconstrução em preço) foi usada como proxy de
"confiança" — a intuição é que um modelo mais certo de uma tendência forte
preveria um retorno de maior magnitude, e essas previsões tenderiam a ser
mais confiáveis que as de magnitude próxima de zero (que podem refletir
apenas ruído/incerteza do modelo). `_confidence_mask` em `src/backtest.py`
calcula o corte de percentil **a partir da própria distribuição de previsões
daquele modelo no teste** (não um valor absoluto fixo, nem compartilhado
entre arquiteturas — decisão já tomada anteriormente e mantida). Testado em
grade: manter as 100% (baseline, sem filtro), 75%, 50%, 25%, 10% e 5%
previsões de maior magnitude, para as 12 combinações mercado×arquitetura×estratégia
(72 execuções via `backtest_sweep()`, salvas em `results/backtest_sweep_log.csv`).

**Por que também um teste de significância estatística:** a métrica de
Sharpe Ratio, sozinha, é enganosa quando o número de trades cai muito (de
~1750 para ~88 no percentil de 5%) — um retorno médio positivo em uma
amostra pequena pode ser puro ruído estatístico, não uma vantagem real
recuperada pelo filtro. Antes de declarar qualquer percentil "vencedor",
`_compute_metrics` agora também roda um teste-t de uma amostra (H0: retorno
médio por trade = 0) sobre a série de retornos de cada execução, reportando
o p-valor (`scipy.stats.ttest_1samp`) junto com Sharpe/Drawdown/VaR.

**Resultado (`results/backtest_sweep_log.csv`, 72 linhas):**

- Das 72 combinações testadas, **41 tiveram resultado estatisticamente
  significativo (p < 0,05)** — e **todas as 41 foram Sharpe negativo**. Ou
  seja: o achado estatisticamente confiável em todo o sweep é que os modelos
  atuais (pré-tuning) perdem dinheiro de forma consistente, não que algum
  percentil de confiança encontrou uma vantagem real.
- As **7 combinações com Sharpe positivo** (ex.: WIN LSTM `single_bar` a 5%
  de confiança, Sharpe +4,06) **todas tiveram p-valor > 0,4** — estatisticamente
  indistinguíveis de retorno médio zero, ou seja, resultado consistente com
  ruído puro em amostras pequenas (88–176 trades). Nenhuma delas é uma
  vantagem real recuperada pelo filtro.
- **Conclusão:** filtrar por magnitude do log-retorno previsto (`|predicted_log_return|`)
  **não** produziu uma vantagem estatisticamente confiável em nenhuma das 12
  combinações mercado/arquitetura/estratégia testadas nesta geração de
  modelos. O Maximum Drawdown cai de forma consistente e mecânica ao reduzir
  o número de trades (menos exposição = menos risco acumulado), mas isso é
  esperado e não indica que o filtro está selecionando trades "melhores" —
  apenas menos trades.

**Interpretação:** magnitude do retorno previsto não se mostrou, nesta
geração de modelos, um proxy confiável de acerto direcional — os modelos
"mais confiantes" não acertam mais que os "menos confiantes" de forma
estatisticamente detectável. Isso é consistente com a acurácia direcional já
observada próxima de 0,50 (decisão 8): se o modelo não tem uma vantagem
direcional real, não há motivo para esperar que suas previsões de maior
magnitude sejam mais confiáveis — magnitude alta pode ser apenas o modelo
"errando com mais convicção".

**Para o TCC:** este é um resultado negativo genuíno e vale reportar como tal
na Seção 3.5 — descarta uma hipótese razoável (seletividade por confiança
resolveria o problema de performance) e aponta o gargalo real para a etapa
seguinte: a acurácia preditiva do modelo em si (tuning de hiperparâmetros),
não a regra de decisão sobre como usar as previsões. Reportar tanto o
resultado (nenhum ganho estatisticamente confiável) quanto o método usado
para verificá-lo (teste-t, não apenas Sharpe bruto) fortalece o rigor
metodológico do capítulo de resultados — é um exemplo concreto de evitar
conclusão precipitada a partir de uma métrica sensível a tamanho de amostra.

### 18. Segunda medida de confiança testada: concordância entre arquiteturas (ensemble)

**A ideia:** já que a magnitude do retorno previsto (decisão 17) não se
mostrou um proxy confiável de acerto, testou-se uma medida de confiança
diferente — concordância de direção entre as 3 arquiteturas (CNN, LSTM,
híbrido) do mesmo mercado. Intuição: se 3 modelos treinados independentemente
(arquiteturas diferentes, mesmos dados) concordam na direção prevista para
uma janela, isso é informação que uma única previsão isolada não carrega, e
poderia indicar um sinal mais robusto que o ruído específico de uma
arquitetura. Implementado em `_load_ensemble_arrays`/`_simulate_strategy_ensemble`
(`src/backtest.py`): opera apenas nas janelas em que as 3 arquiteturas
concordam no sinal de `(previsão - último fechamento)` (unanimidade 3/3, não
maioria 2/3 — filtro mais rigoroso), usando a **média** dos 3 preços
reconstruídos como preço previsto do "ensemble" para aquela operação.
Testado nas 4 combinações mercado×estratégia via `backtest_ensemble_all()`
(`results/backtest_ensemble_log.csv`).

**Resultado:**

| Mercado | Estratégia | Sharpe | Trades (de janelas totais) | p-valor |
|---|---|---|---|---|
| WDO | single_bar | -7,41 | 1057/1753 (60%) | 4,5×10⁻⁷ |
| WDO | multi_bar | -5,91 | 1057/1753 (60%) | 5,4×10⁻⁵ |
| WIN | single_bar | -1,46 | 1443/1759 (82%) | 0,24 |
| WIN | multi_bar | -4,16 | 1443/1759 (82%) | 8,8×10⁻⁴ |

**As 4 combinações têm Sharpe negativo** — nenhuma reverteu para positivo, e
3 das 4 são estatisticamente significativas (p < 0,001), reforçando (não
apenas "não contradizendo") o achado da decisão 17: o problema não é qual
previsão usar dentro do conjunto que o modelo já produz — é que os modelos,
nesta geração, não têm uma vantagem direcional real para nenhuma seleção de
previsões filtrar. A taxa de concordância entre arquiteturas é relativamente
alta (60–82% das janelas), sugerindo que os 3 modelos majoritariamente
concordam entre si — mas concordar não é o mesmo que acertar; é esperado que
modelos treinados nos mesmos dados, com a mesma engenharia de features,
capturem tendências parecidas (inclusive vieses parecidos) mesmo sem terem
uma vantagem preditiva real individualmente.

**Para o TCC:** com dois métodos de seleção de confiança diferentes
(magnitude do retorno previsto e concordância de ensemble) testados e ambos
sem produzir uma vantagem estatisticamente confiável, a conclusão fica mais
robusta para a Seção 3.5: o gargalo desta geração de modelos está na
qualidade da previsão em si, não na regra de decisão sobre quando confiar
nela. Isso justifica, com evidência de duas abordagens independentes, priorizar
tuning de hiperparâmetros do modelo (próxima etapa) em vez de continuar
refinando a camada de estratégia/seleção de trades sobre os modelos atuais.

## Etapa de tuning de hiperparâmetros

Com o backtest baseline confirmando (decisões 12–18) que o gargalo está na
qualidade da previsão em si — não na regra de decisão sobre quando confiar
nela — esta etapa adiciona busca de hiperparâmetros aos 6 modelos, via
`src/tune.py`.

### 19. Objetivo da busca: `val_loss` (MSE), não acurácia direcional ou Sharpe diretamente

A busca de hiperparâmetros otimiza `val_loss` — a mesma métrica que
`EarlyStopping` já monitora em `src/train.py` — e não acurácia direcional ou
métricas de backtest diretamente.

**Por quê:** manter a busca barata e diretamente comparável ao baseline
`_log` (mesmo critério de treino, só variando a configuração). Otimizar
diretamente por acurácia direcional exigiria um laço de validação
customizado (Keras não otimiza isso nativamente via `loss`); otimizar por
Sharpe de backtest exigiria rodar uma simulação a cada trial, o que é caro e,
pelas decisões 17–18, uma métrica ruidosa em amostras pequenas — exatamente
o problema que o teste-t foi introduzido para evitar. RMSE/acurácia
direcional/Sharpe dos modelos vencedores são recalculados depois, via
`evaluate.py`/`backtest.py` apontando para a geração `_tuned`, não durante a
busca.

**Ressalva para o TCC:** um `val_loss` melhor não garante acurácia
direcional ou Sharpe melhores (MSE não diferencia erro para cima de erro
para baixo). Vale reportar os 3 níveis de métrica (erro, direção, negócio)
para os modelos tunados, não só o `val_loss` da busca — a Seção 3.5 já prevê
essa comparação multidimensional.

### 20. Escopo dos hiperparâmetros: arquitetura + treino, exceto `TIMEFRAME`

Estão em escopo: filtros/unidades das camadas Conv1D/LSTM, `kernel_size`,
unidades da camada Dense, taxa de Dropout, `learning_rate` do Adam,
`batch_size` e `patience` do `EarlyStopping`. **Fora de escopo:**
`TIMEFRAME` (janela de lookback, fixa em 60).

**Por quê excluir `TIMEFRAME`:** é o único hiperparâmetro que muda o formato
de entrada do tensor `(timeframe, num_features)`, documentado e sustentado
nas Tabelas 1–3 do artigo (contagem de parâmetros treináveis calculada para
`(60, 5)`). Alterá-lo invalidaria essas tabelas — um custo de retrabalho do
texto já escrito que os outros hiperparâmetros não têm, já que nenhum deles
muda a forma de entrada, só a capacidade/treino do modelo.

**Para o TCC:** se `TIMEFRAME` for revisitado no futuro (ex.: testar janelas
de 30 ou 120 candles), trate como um experimento à parte, com suas próprias
Tabelas 1–3 recalculadas — não substitua silenciosamente o que já foi
escrito, mesma lógica já aplicada à decisão 2 (conjunto de features).

### 21. Método de busca: Keras Tuner `RandomSearch`, 6 buscas independentes (uma por mercado×arquitetura)

A busca usa `keras_tuner.RandomSearch`, com uma busca independente por par
(mercado, arquitetura) — 6 buscas no total, não 3 (uma por arquitetura
compartilhada entre mercados).

**Por quê `RandomSearch` em vez de Hyperband:** com apenas 6 modelos a
tunar, a simplicidade e previsibilidade de custo do `RandomSearch` (N trials
fixos, cada um treinado até `EarlyStopping` decidir parar) pesou mais que o
ganho de eficiência do Hyperband, que exige configurar brackets/fator de
redução corretamente e tem custo de execução menos previsível — não parecia
valer a complexidade adicional para o tamanho deste projeto.

**Por quê buscas separadas por mercado (não só por arquitetura):** a decisão
12 já mostrou que WIN e WDO têm dinâmicas de preço distintas o suficiente
para que uma mesma abordagem de escalonamento quebrasse para um e não para o
outro. Não há garantia de que a mesma capacidade de arquitetura/configuração
de treino seja ótima para os dois — assumir isso e economizar metade das
buscas trocaria uma economia de tempo por um risco de sub-otimizar um dos
dois mercados silenciosamente.

**Implementação:** `src/models.py::create_model_cnn/lstm/hybrid` foram
refatoradas para aceitar um parâmetro opcional `hp` (objeto do Keras Tuner);
quando `hp=None` (uso normal via `train.py`), os valores hardcoded originais
são usados como default — ou seja, os modelos `_log` (decisões 1–11)
continuam reproduzíveis exatamente como antes, sem qualquer mudança de
comportamento. `MarketHyperModel` (`src/tune.py`) conecta essas funções ao
Keras Tuner via `build(hp)`, e também declara `batch_size`/`patience` como
hiperparâmetros dentro de `fit(hp, ...)`, já que o Tuner só varia o que é
declarado via `hp` em algum ponto do ciclo `build`/`fit`.

### 22. Orçamento: 18 trials por busca (~108 treinos no total)

Cada uma das 6 buscas roda até 18 trials (`MAX_TRIALS` em `src/tune.py`),
cada trial treinando até 200 épocas com `EarlyStopping` (mesmo
comportamento de parada antecipada da decisão 4) — na prática, a maioria
dos trials para bem antes de 200, como já observado nos 6 modelos `_log`
(13–34 épocas). Total: até ~108 treinos completos.

**Por quê 18:** intervalo escolhido para permitir exploração real do espaço
de busca (mais que um punhado de tentativas) sem exigir infraestrutura de
GPU dedicada — o orçamento é dimensionado para rodar em CPU/laptop em tempo
de background razoável, dado que os 6 modelos `_log` originais já treinaram
em poucos minutos cada.

### 23. Espaço de busca amplo, centrado nos valores originais

Os ranges pesquisados (ver `src/models.py`) são mais amplos que os valores
originais documentados no artigo, não apenas uma vizinhança estreita deles:

- Conv1D filtros: camada 1 em `{32, 64, 128, 256}`, camada 2 em
  `{16, 32, 64, 128}` (original: 64 e 32).
- `kernel_size`: `{2, 3, 4, 5, 7}` (original: 3).
- LSTM unidades: camada 1 em `{32, 64, 96, 128}`, camada 2 em
  `{16, 32, 64}` (original: 64 e 32); no híbrido, unidade única em
  `{32, 50, 64, 96}` (original: 50).
- Dense: `{16, 25, 32, 64}` (original: 25).
- Dropout: 0,1 a 0,5, passo 0,05 (original: 0,2 fixo).
- `learning_rate`: 1×10⁻⁴ a 1×10⁻², escala logarítmica (original: 1×10⁻³,
  default do Adam, nunca declarado explicitamente antes).
- `batch_size`: `{16, 32, 64}` (original: 32).
- `patience`: `{8, 12, 16}` (original: 12).

**Por quê um espaço amplo:** a intenção é dar à busca uma chance real de
encontrar uma configuração melhor, mesmo que fora da vizinhança imediata do
que já foi tentado manualmente — dado que o baseline atual (decisões 1–18)
já mostrou desempenho fraco (acurácia direcional ~0,50, nenhuma vantagem de
Sharpe estatisticamente confiável), não há motivo para supor que a
arquitetura/configuração original já esteja próxima do ótimo.

**Trade-off reconhecido:** modelos no extremo superior do espaço (ex.: 256
filtros + 128 unidades LSTM) têm uma contagem de parâmetros muito maior que
a documentada nas Tabelas 1–3 do artigo. Se a configuração vencedora usar
parâmetros bem acima do original, as Tabelas 1–3 (que descrevem
especificamente a arquitetura original/baseline) devem ser mantidas como
estão — descrevem o modelo antes do tuning — e uma tabela nova, análoga,
deve ser adicionada para a arquitetura vencedora tunada, com sua própria
contagem de parâmetros. Ver decisão 2 para o mesmo princípio já aplicado.

### 24. Artefatos: geração `_tuned`, hiperparâmetros persistidos em JSON

Os hiperparâmetros vencedores de cada busca são salvos em
`models/{market}/{architecture}_best_hp.json`. O modelo final é então
retreinado (não apenas o melhor trial da busca, mas um treino completo e
determinístico com esses hiperparâmetros fixos) e salvo como
`models/{market}/{architecture}_tuned.keras` — nova geração de artefatos,
seguindo o mesmo padrão de coexistência sem sobrescrita já usado entre as
gerações original e `_log` (decisão 12). `evaluate.py`/`backtest.py` já
suportam essa troca de geração via seu parâmetro `suffix` existente
(`suffix='_tuned'`), sem precisar de mudanças adicionais nesses dois
módulos.

**Por quê retreinar em vez de reaproveitar o modelo do melhor trial:** o
Keras Tuner reinicia os pesos a cada trial por padrão; o objetivo da busca é
encontrar a *configuração*, não necessariamente preservar aquele
treinamento específico. Retreinar com `train_one_model` garante que o
processo de treino final do modelo tunado seja idêntico em estrutura ao dos
outros dois (original e `_log`) — mesma função, mesmo fluxo de salvamento —
diferindo apenas na configuração de hiperparâmetros usada.

**Diretório de scratch do Keras Tuner** (`tuning/`, criado por
`kt.RandomSearch(directory=...)` com os logs/checkpoints intermediários de
cada trial) foi adicionado ao `.gitignore` — não é um artefato final, o
resultado que importa já está persistido em `_best_hp.json` e
`{architecture}_tuned.keras`.

**Para o TCC:** ao reportar os resultados de tuning (Seção 3.5 ou uma nova
subseção de Análises), inclua a tabela de hiperparâmetros vencedores por
(mercado, arquitetura) — direto de `models/{market}/{architecture}_best_hp.json`
— e a comparação de métricas `_log` vs. `_tuned` (RMSE, MAE, acurácia
direcional, Sharpe) lado a lado, não apenas os hiperparâmetros isolados.

### 25. Resultado do tuning: terceira intervenção sem ganho estatisticamente confiável

A busca completa (`python -m src.tune`, 18 trials × 6 pares, ~108 treinos)
rodou até o fim (~6 horas). Os hiperparâmetros vencedores por par estão em
`models/{market}/{architecture}_best_hp.json`; a comparação completa
`_log` vs. `_tuned` está em `results/comparison_log.csv` /
`results/comparison_tuned.csv` (métricas de erro) e
`results/backtest_comparison_log.csv` / `results/backtest_comparison_tuned.csv`
(métricas de negócio).

**Resultado — métricas de erro (`evaluate.py`):**

- Acurácia direcional permanece praticamente inalterada, ~0,49–0,50 nos 6
  modelos, antes e depois do tuning — sem sinal de que a busca encontrou uma
  configuração com vantagem direcional real.
- RMSE mudou pouco e sem direção consistente: a maior melhora foi WDO/LSTM
  (11,99 → não, especificamente 13,85 → 11,98, uma queda de ~1,87), mas
  WDO/Híbrido piorou (11,99 → 13,66, alta de ~1,68) — variação compatível
  com ruído de treino, não com uma melhora sistemática de capacidade.

**Resultado — métricas de negócio (`backtest.py`):**

- **Sharpe Ratio permanece negativo nas 12 combinações mercado×arquitetura×estratégia,
  tanto antes quanto depois do tuning.** 10/12 combinações já eram
  estatisticamente significativas (p < 0,05) na geração `_log`; a geração
  `_tuned` tem 11/12 significativas — ou seja, o achado confiável (perda
  consistente) ficou **mais** robusto, não menos.
- Algumas combinações individuais melhoraram (ex.: WDO/CNN `multi_bar`:
  Sharpe -7,37 → -4,57; WIN/Híbrido `multi_bar`: -5,46 → -4,60), outras
  pioraram (ex.: WIN/CNN `single_bar`: -1,12 → -1,89) — sem padrão
  consistente de melhora, o que é o comportamento esperado de variação por
  ruído de hiperparâmetros, não de uma melhoria real de capacidade
  preditiva.
- Efeito colateral notado (não é um bug): após o tuning, os 3 modelos de WDO
  passaram a produzir exatamente os mesmos resultados de backtest entre si
  (Sharpe/Drawdown/VaR idênticos por estratégia). Verificado que as previsões
  numéricas continuam diferentes entre arquiteturas (diferença de até ~7,26
  pontos), mas o **sinal direcional** (a única coisa que o backtest usa)
  coincide em 100% das janelas entre CNN e LSTM — mesmo fenômeno já
  observado e documentado para dois dos três modelos WDO da geração `_log`
  (ver `NOTAS.md`, seção de backtest), apenas mais pronunciado após o
  tuning. Não indica extrapolação de escala (o diagnóstico da decisão 12 foi
  verificado como não aplicável ao WDO nesta divisão de dados) — é apenas
  coincidência de sinal entre modelos que aprenderam tendências parecidas.

**Conclusão — terceira intervenção independente sem ganho:** com filtragem
por confiança (decisão 17), concordância de ensemble (decisão 18) e agora
tuning de hiperparâmetros (arquitetura + configuração de treino, espaço de
busca amplo) todas testadas rigorosamente (com teste-t, não apenas Sharpe
bruto) e nenhuma produzindo uma vantagem estatisticamente confiável, o
padrão que emerge é consistente: o gargalo não está na regra de decisão
sobre quando confiar no modelo (decisões 17–18) nem na capacidade/configuração
da arquitetura (decisão 25) — está na informação disponível nas features de
entrada em si. As 5 features atuais (OHLC em log-retorno + volume bruto,
decisão 12) parecem não conter sinal direcional suficiente para essas 3
arquiteturas de Deep Learning, neste horizonte de previsão (1 hora), para
superar o acaso.

**Para o TCC:** esta é a conclusão mais forte e melhor evidenciada do
capítulo de Análise de Resultados até agora — não por ausência de esforço
metodológico, mas porque três abordagens independentes e complementares
(seleção de trades por confiança, seleção por consenso entre arquiteturas, e
otimização da capacidade do modelo) convergem para o mesmo diagnóstico.
Vale apresentar as três tentativas em sequência na Seção 3.5, com seus
respectivos testes de significância, como evidência cumulativa — não como
três experimentos desconectados, mas como um processo de eliminação
sistemática de hipóteses que aponta consistentemente para a mesma causa
raiz. O próximo teste natural (engenharia de atributos — indicadores
técnicos como RSI, médias móveis e MACD, já citados na Seção 2 do artigo
como potencialmente úteis [Mohammed 2022]) ataca exatamente essa causa raiz,
ao invés de repetir uma abordagem já descartada.

## Etapa de engenharia de atributos: indicadores técnicos

Com três intervenções independentes (filtragem por confiança, concordância
de ensemble, tuning de hiperparâmetros) apontando para a mesma causa raiz —
as 5 features atuais não carregam sinal direcional suficiente —, esta etapa
ataca diretamente essa causa, adicionando indicadores técnicos como
features de entrada extras, seguindo a motivação já citada na Seção 2 do
artigo [Mohammed 2022].

### 26. Nova geração `_features`: RSI, SMA (log-retorno) e MACD (normalizado), 10 features totais

**Escopo:** esta é uma geração de modelos explicitamente nova e separada
(`suffix='_features'`), não uma substituição das gerações `_log`/`_tuned`.
Motivo: adicionar indicadores muda a contagem de colunas de entrada de 5
para 10 — alterar o formato `(60, 5)` invalidaria a comparação já registrada
nas Tabelas 1–3 do artigo, o mesmo cuidado já tomado nas decisões 2 e 20 ao
proteger esse formato de outras mudanças de escopo. As 5 features base (OHLC
em log-retorno + volume bruto) permanecem exatamente como na decisão 12;
apenas 5 colunas novas são adicionadas.

**Indicadores escolhidos:** os três citados no Referencial Teórico do artigo
[Mohammed 2022] — RSI, uma média móvel e MACD — implementados via
`pandas_ta` (`src/features.py::add_technical_indicators`) em vez de
calculados manualmente, para evitar erros sutis de implementação (RSI em
particular tem uma formulação de suavização — Wilder's smoothing — fácil de
implementar errado à mão).

**Períodos usados (mais curtos que o padrão de gráfico diário):**
- RSI: 7 períodos (padrão de mercado: 14).
- SMA: 10 períodos (padrão comum: 20).
- MACD: rápida 6 / lenta 13 / sinal 4 (padrão de mercado: 12/26/9).

**Por quê períodos mais curtos:** os defaults citados na literatura de
mercado (RSI-14, SMA-20, MACD 12/26/9) foram calibrados historicamente para
gráficos diários. Aplicados a candles de 1 hora com uma janela de entrada de
apenas 60 barras (60 horas ≈ 7-8 pregões), um RSI-14 ou MACD-26 exigiria
quase metade da janela de entrada só para "aquecer" o indicador, sobrando
pouco espaço para o modelo ver o indicador reagir a mudanças de mercado
dentro da própria janela. Períodos proporcionalmente mais curtos mantêm os
indicadores responsivos dentro do horizonte de 60 barras usado pelo pipeline.

**Tratamento de escala por indicador (evitando repetir o problema da decisão
12):**
- **RSI:** já é um oscilador limitado a [0, 100] por construção — mantido em
  nível bruto. Sem risco de extrapolação, pois o intervalo é fixo
  independente de quanto o preço subiu ou desceu.
- **SMA:** é uma série de nível de preço (assim como open/high/low/close),
  com o mesmo risco de deriva de escala diagnosticado na decisão 12 —
  convertida para log-retorno da própria SMA (`log(SMA_t / SMA_{t-1})`) pelo
  mesmo motivo e com a mesma garantia de estacionariedade.
- **MACD (linha, sinal, histograma):** por ser uma diferença de médias
  exponenciais, já é mais próximo de estacionário que o preço bruto, mas
  ainda pode variar em magnitude absoluta com o nível de preço ao longo de
  uma tendência longa. Normalizado dividindo cada uma das 3 séries pelo
  preço de fechamento (`MACD / close`, etc.), tornando a magnitude
  comparável entre os regimes de preço do treino e do teste mesmo com
  deriva de tendência de longo prazo — mesma preocupação estrutural da
  decisão 12, resolvida aqui de forma diferente (normalização pelo preço em
  vez de log-retorno, mais apropriado para uma série que já é uma diferença,
  não um nível).

**Alinhamento de linhas:** `add_technical_indicators` descarta o período de
aquecimento dos indicadores (mais linhas que o único descarte de
`prepare_features`, já que RSI/SMA/MACD precisam de várias barras de
histórico antes do primeiro valor válido). `load_prepared` aplica esse
descarte tanto no DataFrame de features quanto no DataFrame de preços reais,
na mesma ordem, para os índices continuarem alinhados entre os dois — o
mesmo cuidado já presente no tratamento da geração `_log`.

**Refatoração de `train.py`/`evaluate.py`/`backtest.py`:** os três módulos
passaram a aceitar um parâmetro `feature_set` (`'base'` ou `'extended'`),
paralelo ao `suffix` já existente. `FEATURE_COLS`/`CLOSE_COL_IDX` viraram
`FEATURE_SETS` (um dicionário `{'base': [...], 'extended': [...]}`) em
`train.py`, importado por `evaluate.py`/`backtest.py` para dimensionar
corretamente os arrays de dummy usados na reversão de escala. `feature_set`
deve corresponder ao conjunto realmente usado no treino do `suffix` escolhido
— chamar `evaluate_one_model(..., suffix='_features', feature_set='base')`
por engano produziria um mismatch silencioso de formato de array. Optou-se
por essa abordagem (parametrizar os módulos existentes) em vez de duplicar a
orquestração em um módulo isolado, para reaproveitar toda a lógica de
métricas/backtest já validada nas gerações anteriores sem duplicação de
código.

**Nova dependência: `pandas-ta`.** Ao instalar, o `pip` rebaixou o `numpy`
de 2.4.6 para 2.2.6 para satisfazer a dependência transitiva `numba`
(usada pelo `pandas-ta` para compilação JIT de alguns indicadores).
Verificado que TensorFlow, pandas e scikit-learn continuam funcionando
corretamente nessa versão do numpy antes de prosseguir — ver `CLAUDE.md`
para o registro dessa mudança de versão, caso o ambiente pareça
"rebaixar" o numpy inesperadamente no futuro.

**Para o TCC:** ao descrever esta etapa na Seção 3.2/3.5, cite explicitamente
[Mohammed 2022] como motivação já presente na revisão de literatura do
próprio artigo — reforça que esta não é uma tentativa aleatória, mas o
próximo passo lógico indicado pela própria fundamentação teórica do
trabalho. Uma tabela de arquitetura própria (análoga às Tabelas 1–3, mas
para o formato de entrada `(60, 10)`) deve ser incluída para os 3 modelos
desta geração, com sua contagem de parâmetros treináveis recalculada — não
reaproveitar as Tabelas 1–3 originais para esta variação.

### 27. Resultado da geração `_features`: quarta intervenção sem ganho estatisticamente confiável

Os 6 modelos da geração `_features` foram treinados, avaliados e testados em
backtest (`results/comparison_features.csv`,
`results/backtest_comparison_features.csv`).

**Métricas de erro:** acurácia direcional permanece em ~0,49–0,50 nos 6
modelos — essencialmente idêntica às gerações `_log` e `_tuned`, sem
qualquer sinal de que RSI/SMA/MACD tenham adicionado informação direcional
que os 3 modelos conseguissem explorar.

**Métricas de negócio:** Sharpe Ratio permanece **negativo nas 12
combinações** mercado×arquitetura×estratégia, 11 das 12 estatisticamente
significativas (p < 0,05) — o mesmo padrão das gerações anteriores, apenas
confirmado novamente com um conjunto de features diferente.

**Conclusão — quarta intervenção independente sem ganho:** com filtragem
por confiança (decisão 17), concordância de ensemble (decisão 18), tuning
de hiperparâmetros (decisão 25) e agora engenharia de atributos com
indicadores técnicos (decisão 27) todas testadas e nenhuma produzindo uma
vantagem estatisticamente confiável, o padrão de eliminação sistemática de
hipóteses (já descrito na decisão 25) fica ainda mais robusto. Nem a
capacidade do modelo, nem a seleção de quando confiar nele, nem o conjunto
de features de entrada testado (incluindo os indicadores citados na própria
revisão de literatura do artigo) produziram uma vantagem direcional
detectável para WIN/WDO neste horizonte de 1 hora.

**Para o TCC:** mantém a mesma recomendação da decisão 25 — reportar como
achado cumulativo, não como falha isolada. A tabela de arquitetura
`(60, 10)` mencionada acima e a comparação completa `_log`/`_tuned`/`_features`
(erro + negócio) devem constar juntas na Seção 3.5.

## Etapa final: filtro de confiança combinado com tuning e engenharia de atributos

Com quatro intervenções independentes (decisões 17, 18, 25, 27) sem produzir
uma vantagem estatisticamente confiável, testou-se uma última combinação:
aplicar o filtro de confiança por magnitude (decisão 17, o único dos quatro
que já havia produzido pelo menos um Sharpe positivo, ainda que não
significativo) sobre as gerações `_tuned` e `_features`, em vez de apenas
sobre `_log`. A pergunta: um modelo mais bem calibrado (tuning) ou com mais
informação de entrada (indicadores técnicos), quando também filtrado para
operar só nas previsões de maior confiança, produz uma vantagem que resista
a um teste de significância?

### 28. Resultado: um achado aparente que não sobrevive à correção por comparações múltiplas

`backtest_sweep()` foi rodado sobre `_tuned` e `_features` (mesma grade de
percentis da decisão 17: 100/75/50/25/10/5%), somando-se ao sweep já
existente sobre `_log`. Resultado agregado das 3 gerações (216 combinações
testadas ao todo: 72 por geração × 3 gerações):

| Geração | Combinações com Sharpe > 0 | Combinações significativas (p<0,05) | Significativas E positivas |
|---|---|---|---|
| `_log` | 7 / 72 | 41 / 72 | 0 |
| `_tuned` | 10 / 72 | 41 / 72 | 0 |
| `_features` | 14 / 72 | 43 / 72 | **3** |

As 3 combinações significativas e positivas, todas na geração `_features`,
todas no percentil mais extremo (5% de confiança, 88 trades):

| Mercado | Arquitetura | Estratégia | Sharpe | p-valor | Trades |
|---|---|---|---|---|---|
| WDO | Híbrido | multi_bar | +12,33 | 0,0168 | 88 |
| WDO | CNN | multi_bar | +11,86 | 0,0213 | 88 |
| WIN | LSTM | multi_bar | +10,47 | 0,0413 | 88 |

**Por que isso não é o achado que parece ser à primeira vista — correção
por comparações múltiplas:** ao todo, 216 testes de hipótese independentes
foram rodados entre as 3 gerações (72 por geração). Com um limiar de
significância de 5% (`p < 0,05`), o número **esperado de falsos positivos
por puro acaso**, mesmo que nenhuma das 216 combinações tivesse qualquer
vantagem real, é `216 × 0,05 ≈ 10,8`. Encontrar 3 combinações com
`p < 0,05` está bem dentro do que o acaso sozinho produziria ao testar 216
hipóteses — não é evidência de que essas 3 combinações específicas sejam
especiais.

Aplicando a correção de Bonferroni (o método mais simples e conservador para
esse problema: dividir o limiar de significância pelo número de testes
realizados, aqui `0,05 / 216 ≈ 0,000231`), **nenhuma das 3 combinações
sobrevive**: os p-valores observados (0,0168 / 0,0213 / 0,0413) estão
todos ordens de grandeza acima do limiar corrigido. Ou seja, mesmo o
resultado "mais promissor" encontrado em toda a extensão dos experimentos
deste projeto (tuning + features + filtro de confiança, em 3 gerações e 216
combinações testadas) não resiste ao ajuste estatístico apropriado para a
quantidade de buscas realizadas.

**Padrão adicional que reforça essa leitura:** as 3 combinações
significativas concentram-se exatamente no percentil de confiança mais
extremo testado (5%, a menor amostra possível na grade — 88 trades), o
mesmo padrão já observado na decisão 17 para os "melhores" resultados de
`_log`/`_tuned` (Sharpe alto, mas não significativo). É o comportamento
esperado de **ruído de amostra pequena com variância alta**: quanto menor a
amostra, maior a chance de um Sharpe extremo por acaso, positivo ou
negativo — e a grade de busca testou justamente esse extremo em 3 gerações
diferentes, aumentando a chance de encontrar um "acerto" por puro volume de
tentativas, não por sinal real.

**Conclusão final — nenhuma combinação testada produziu uma vantagem
direcional confiável.** Com controle rigoroso para múltiplas comparações,
o resultado permanece consistente com as quatro intervenções anteriores:
nenhuma combinação de arquitetura, hiperparâmetros, conjunto de features ou
regra de seleção de trades testada neste projeto produziu uma vantagem
estatisticamente defensável para prever a direção do preço de WIN/WDO em
um horizonte de 1 hora usando CNN, LSTM ou o híbrido CNN-LSTM sobre dados
de candle H1.

**Para o TCC — como reportar isso na Seção 3.5:** este é o ponto de parada
natural da fase de experimentação de modelagem, e o resultado é forte
precisamente por ser negativo de forma tão bem controlada:

1. Apresentar a tabela de 5 intervenções em sequência (filtro de confiança
   por magnitude, concordância de ensemble, tuning de hiperparâmetros,
   indicadores técnicos, e a combinação de tuning/features com filtro de
   confiança) como um processo metodológico de eliminação de hipóteses, não
   como experimentos avulsos.
2. Destacar explicitamente o cuidado com comparações múltiplas como um
   diferencial metodológico do trabalho — é comum em pesquisas de
   trading algorítmico "encontrar" uma estratégia vencedora que na
   verdade é um artefato de testar muitas variações e reportar só a
   melhor (data snooping / p-hacking), e este projeto documenta
   explicitamente ter verificado e descartado essa possibilidade.
3. A conclusão honesta — "para as arquiteturas, features e regras de decisão
   testadas, não há evidência estatisticamente robusta de vantagem
   direcional prevísivel neste horizonte" — é compatível com a literatura
   de eficiência de mercado (a hipótese de que preços de curtíssimo prazo
   incorporam informação disponível rapidamente, dificultando previsão
   sistemática) e serve como uma conclusão legítima e defensável para a
   pergunta de pesquisa do artigo, mesmo não sendo o resultado
   "esperado"/desejado inicialmente.
4. Não é necessário (nem honesto) apresentar as 3 combinações com
   `p < 0,05` da geração `_features` como um resultado positivo — se
   mencionadas, devem vir acompanhadas da análise de correção por
   comparações múltiplas acima, para não induzir a banca a uma conclusão
   que os dados não sustentam.
