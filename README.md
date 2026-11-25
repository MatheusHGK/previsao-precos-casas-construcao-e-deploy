# Previsão de preços de imóveis — King County (Seattle)

Projeto de ponta a ponta para estimar o preço de venda de imóveis residenciais em King County (Seattle, EUA). Cobre a análise exploratória, o treinamento e a interpretação do modelo, o deploy em produção no Databricks e a estratégia de monitoramento e retreino.

O enunciado original do case está em [`ENUNCIADO.md`](ENUNCIADO.md).

---

## Resultados principais

| Item | Valor |
|---|---|
| Modelo campeão | CatBoost — base `v0_base`, alvo `log_price` |
| Hiperparâmetros | 620 árvores, profundidade 8, learning rate 0,05 |
| MAPE (treino / validação cruzada / holdout) | 9,27% / 11,34% / 11,50% |
| Holdout — RMSE / MAE / R² | US$ 118.358 / US$ 61.901 / 0,909 |
| Imóveis com erro até ±10% / ±20% | 58,8% / 84,5% |
| Latência do endpoint | ~1 s por requisição |

O erro de validação cruzada (11,34%) e o do holdout (11,50%) são próximos, o que indica boa generalização para imóveis não vistos no treino.

Limitação conhecida: por ser um modelo de árvores, ele não extrapola além dos valores vistos no treino. O teto de previsão fica próximo de US$ 5 milhões (o máximo do treino é US$ 7,7 milhões), há viés de superestimação nos imóveis mais baratos (+17% no primeiro decil) e de subestimação nos mais caros (−7% no último decil). Por isso, as faixas de cada variável usadas no treino são salvas em `metadados.json` e usadas como alerta de faixa.

---

## Estrutura do repositório

```
├── README.md
├── ENUNCIADO.md                         # Enunciado original do case
├── requirements.txt
├── kc_house_data.csv                    # Base original (imóveis)
├── zipcode_demographics.csv             # Base original (demografia por CEP)
├── future_unseen_examples.csv           # Imóveis novos a prever
├── *_traduzido.csv                      # Bases com colunas traduzidas (geradas na EDA)
├── kc_house_treino.csv                  # Base de treino limpa (gerada na EDA)
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Treinamento e engenharia reversa.ipynb
│   └── Deploy_Modelo_CatBoost_Precificacao_Imoveis_-_Databricks.ipynb
│
├── Apresentacoes/
│   ├── Previsão de preços de imóveis - Apresentação para time de negócio.pdf
│   └── Desenho da arquitetura - deploy do modelo Online e Offline.png
│
├── Modelo final/
│   ├── modelo_final.joblib              # Modelo CatBoost treinado
│   ├── metadados.json                   # Features, hiperparâmetros, métricas e faixas do treino
│   ├── holdout_previsoes.csv            # Previsões do holdout
│   ├── unseen_previsoes.csv             # Previsões dos imóveis novos (gerado no deploy)
│   └── unseen_previsoes_local.csv       # Mesmas previsões geradas localmente, para comparação
│
├── Resultados parametrizacao v3/
│   ├── resultados.csv                   # Métricas de todos os modelos testados
│   ├── curvas_arvores.csv
│   ├── curvas_profundidade.csv
│   └── graficos/
│
└── Variacoes da base de treino/         # Variações v0_base a v7 usadas na comparação
```

---

## Como executar

### Ambiente

- Python 3.13
- Java (JDK 17), necessário para o PySpark usado na EDA
- Dependências: `pip install -r requirements.txt`

### Caminho do projeto

Os notebooks de EDA e de treinamento usam a variável `PASTA`, definida na primeira célula de configuração, com o caminho local do repositório. Antes de executar, essa variável deve ser ajustada para o caminho da máquina.

### Ordem dos notebooks

1. **`EDA.ipynb`** — executado localmente. Lê as bases originais e gera as bases traduzidas e a `kc_house_treino.csv`.
2. **`Treinamento e engenharia reversa.ipynb`** — executado localmente. Compara modelos e variações da base, faz a parametrização, avalia no holdout e interpreta o modelo (importância de variáveis e SHAP).
3. **`Deploy_Modelo_CatBoost_Precificacao_Imoveis_-_Databricks.ipynb`** — executado **somente no Databricks** (ver abaixo).

### Observações sobre reprodutibilidade

- O treinamento foi executado em etapas, com checkpoints salvos em disco, por isso a numeração de execução das células não é sequencial.
- As bases em `Variacoes da base de treino/` e o arquivo `holdout_previsoes.csv` são entregues prontos e são lidos pelos notebooks.
- O arquivo `unseen_previsoes_local.csv` foi gerado carregando o `modelo_final.joblib` localmente e serve apenas para conferir que o modelo em produção devolve as mesmas previsões.

### Notebook de deploy (Databricks)

O notebook de deploy usa MLflow, Unity Catalog e Model Serving, e por isso só executa dentro de um workspace Databricks. Fora dele o notebook pode ser lido normalmente, mas alguns outputs do MLflow aparecem como "Could not render content".

Para reproduzir, é necessário:

- **Volume** com os arquivos: `modelo_final.joblib`, `metadados.json`, `zipcode_demographics.csv`, `future_unseen_examples.csv` e `unseen_previsoes_local.csv`;
- **Tabelas no Unity Catalog** (`workspace.case_madeinweb`): `kc_house_treino`, `zipcode_demographics` e `holdout_previsoes`.

### Teste rápido do endpoint

A primeira célula do notebook de deploy chama o endpoint em produção via `requests` e pode ser executada em qualquer ambiente Python. O token de acesso não está no repositório: ele é enviado separadamente ao avaliador e deve ser colocado no lugar de `<TOKEN_DE_ACESSO>`. O token será revogado após a avaliação.

---

## 1. Análise exploratória (`EDA.ipynb`)

- Leitura das bases com PySpark e tradução das colunas para português.
- Análise das distribuições, correlações com o preço e outliers.
- União da base física dos imóveis com a base demográfica por CEP (`zipcode`).
- Análise geográfica: os imóveis mais caros se concentram na faixa central (Seattle, Bellevue, Medina e Mercer Island).
- Geração da base de treino limpa (`kc_house_treino.csv`).

## 2. Modelagem (`Treinamento e engenharia reversa.ipynb`)

- **Holdout fixo** de 20%, definido por hash do `id` (`xxhash64(id) % 100 < 20`), para garantir que o mesmo imóvel caia sempre no mesmo conjunto.
- **Validação cruzada** K-fold (k = 5, seed 42) sobre os 80% restantes.
- **Modelos comparados**: LightGBM, XGBoost e CatBoost (e GBR nas rodadas iniciais).
- **Variações da base**: 9 variações (`v0_base` a `v7`) testando diferentes conjuntos de variáveis derivadas.
- **Alvos testados**: `log_price` e `log_price_sqft`, entre outros (4 alvos avaliados na `v0_base`).
- **Parametrização**: um parâmetro por vez. Primeiro a curva de número de árvores, com regra de parada (melhora mínima de 0,08%, paciência de 10 e máximo de 1.000 árvores) e escolha pelo cotovelo da curva; depois a profundidade de 3 a 10 (melhora mínima de 0,25%, paciência de 2).
- **Engenharia reversa**: importância das variáveis, valores SHAP e análise de erro por faixa de preço (decis), que expôs a limitação de extrapolação descrita acima.

## 3. Deploy

### Arquitetura

![Arquitetura do deploy](Apresentacoes/Desenho%20da%20arquitetura%20-%20deploy%20do%20modelo%20Online%20e%20Offline.png)

### Camadas

**Modelo empacotado (MLflow pyfunc).** O modelo é encapsulado em um wrapper `HousePriceModel` que recebe somente os dados do imóvel. Dentro do wrapper são feitos o join com a base demográfica por CEP (embutida como artefato do modelo) e o cálculo das variáveis derivadas (`house_age`, `was_renovated`, `yr_last_update`). Assim, quem consome a API não precisa conhecer as features internas do modelo. O artefato é validado em ambiente isolado (`uv`) antes do registro.

**API (Model Serving).** O modelo é servido no endpoint `modelo-precificacao-imoveis` do Databricks Model Serving (CPU Small, com scale-to-zero), com latência de cerca de 1 s. O endpoint recebe o JSON do imóvel e devolve `preco_previsto` e `log_price_previsto`. Como alternativa fora do Databricks, a mesma estrutura pode ser servida em Kubernetes (AKS) com FastAPI e Docker, carregando o modelo do MLflow.

**Fluxo batch (offline).** Para previsões em lote: arquivo enviado no SharePoint → Logic Apps → job no Databricks → merge em tabela Delta → resultado disponível em CSV e para consulta via Genie.

**Versionamento.** O modelo é registrado no Unity Catalog (`workspace.case_madeinweb.modelo_precificacao_imoveis`, versão atual 5), com métricas, parâmetros, assinatura de entrada e faixas do treino. A troca de versão é feita por alias (campeão/desafiante), o que permite rollback imediato para a versão anterior.

**Monitoramento.** Acompanhamento do drift das variáveis de entrada (PSI), do percentual de requisições com variáveis fora da faixa do treino, do viés das previsões e, quando as vendas são concluídas, do MAPE real.

### Validação do deploy

As previsões do modelo em produção foram comparadas às do modelo treinado localmente: diferença máxima de 2,08e-16 no holdout e de 2,09e-16 nos imóveis novos, ou seja, os resultados são idênticos.

## 4. Aprendizado contínuo

- **Monitoramento imediato e tardio**: o preço real de venda só é conhecido semanas depois da previsão. Por isso, drift de entrada (PSI) e viés das previsões funcionam como sinais antecipados, e o MAPE real confirma depois.
- **Gatilhos de retreino**: PSI > 0,2 em variáveis importantes, viés acima de ±3% ou MAPE acima de 13% por 2 meses seguidos.
- **Cadência**: retreino leve mensal (mesmos hiperparâmetros, dados novos) e revisão completa trimestral ou por alerta, repetindo a fábrica de modelos com parametrização por K-fold.
- **Janela de dados**: janela móvel, priorizando as vendas mais recentes; atualização da base demográfica quando sai um novo censo.
- **Avaliação e substituição**: o modelo desafiante é comparado ao campeão em um holdout temporal (vendas mais recentes). Se for melhor, roda em modo shadow (recebe as mesmas requisições sem responder ao usuário) antes de virar campeão. Em caso de piora, rollback para a versão anterior pelo alias.

## 5. Comunicação com stakeholders

A apresentação para o time de negócio está em [`Apresentacoes/`](Apresentacoes/Previsão%20de%20preços%20de%20imóveis%20-%20Apresentação%20para%20time%20de%20negócio.pdf). Ela traduz os resultados em linguagem de negócio: precisão do modelo, comparação com referências de mercado, fatores que mais influenciam o preço, limitações, recomendação de uso e próximos passos.

---

## Previsão dos imóveis novos

Os 100 imóveis de `future_unseen_examples.csv` foram previstos pelo modelo em produção (seção 5.1 do notebook de deploy). Nenhum deles possui variáveis fora da faixa do treino. O resultado está em `Modelo final/unseen_previsoes.csv` e na tabela `workspace.case_madeinweb.unseen_previsoes`.

---

## Próximos passos

- Modelos segmentados por faixa de preço, para reduzir o viés nos extremos.
- Inclusão do intervalo de previsão e do alerta de faixa na resposta da API (hoje o endpoint retorna apenas o preço previsto).
