# Databricks Banking Fraud Detection MLOps Pipeline

Projeto completo de MLOps para detecção de fraudes bancárias usando Databricks, Delta Lake, Unity Catalog, MLflow, Feature Engineering, Data Quality, Drift Monitoring, Databricks Asset Bundles e Model Serving.

O objetivo é demonstrar uma arquitetura reprodutível de Machine Learning em múltiplos ambientes, cobrindo desde a ingestão do dataset até batch inference e inferência online por API REST.

---

## Status do projeto

| Ambiente | Pipeline | Modelo | Batch inference | Endpoint | API REST |
|---|---:|---:|---:|---:|---:|
| DEV | Concluído | Registrado | Validada | READY | Validada |
| ACC | Concluído | Registrado | Validada | READY | Validada |
| PROD | Concluído | Registrado | Validada | READY | Validada |

### Resultado atual

- Melhor modelo: `RandomForest`
- F1 Score: `0.824242`
- Recall: `0.715789`
- Registros Bronze: `284.807`
- Registros Silver, Gold e Features: `283.726`
- Registros de batch inference: `283.726`
- Data Quality: `0` registros em quarentena
- Drift: `34 OK`, `0 WARNING`, `0 DRIFT`
- Serving DEV: `mlops_dev.fraud.fraud_detection_serving_model`, versão `4`
- Serving ACC: `mlops_acc.fraud.fraud_detection_serving_model`, versão `1`
- Serving PROD: `mlops_production.fraud.fraud_detection_serving_model`, versão `2`
- Endpoint PROD: `fraud-detection-endpoint-prod`
- Estado PROD: `READY / DEPLOYMENT_READY`
- API REST PROD: `HTTP 200`
- Cliente Python PROD: validado
- Evidências PROD: salvas em `serving/results/`

---

## Objetivo

Construir uma esteira completa de MLOps para classificação de transações fraudulentas, contemplando:

- ingestão de dados;
- arquitetura Bronze, Silver e Gold;
- validações de Data Quality;
- Feature Engineering;
- Feature Table no Unity Catalog;
- treinamento de múltiplos algoritmos;
- rastreabilidade de experimentos no MLflow;
- seleção automática do melhor modelo;
- Model Registry no Unity Catalog;
- batch inference;
- monitoramento de drift;
- decisão automática de retreinamento;
- gate de aprovação;
- champion/challenger;
- rollback automático;
- Model Serving;
- API REST;
- promoção entre DEV, ACC e PROD;
- CI/CD com Databricks Asset Bundles e GitHub Actions.

---

## Arquitetura

```text
Kaggle Credit Card Fraud Dataset
              |
              v
Unity Catalog Volume
              |
              v
       Bronze Delta Table
              |
              v
       Silver Delta Table
              |
       +------+------+
       |             |
       v             v
 Data Quality    Quarantine
       |
       v
        Gold Delta Table
              |
              v
       Feature Engineering
              |
              v
 Unity Catalog Feature Table
              |
              v
       Multi-model Training
              |
              v
          MLflow Tracking
              |
              v
      Best Model Selection
              |
       +------+------+
       |             |
       v             v
 Batch Inference   Model Registry
       |             |
       v             v
 Delta Predictions Model Serving
                     |
                     v
                  REST API
```

---

## Tecnologias utilizadas

### Databricks e Lakehouse

- Databricks Workspace
- Delta Lake
- Unity Catalog
- Databricks Volumes
- Databricks Asset Bundles
- Databricks Jobs
- Databricks Model Serving
- Serverless Compute

### Machine Learning e MLOps

- MLflow Tracking
- MLflow Model Registry
- MLflow PyFunc
- Scikit-learn
- Random Forest
- Logistic Regression
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Champion/Challenger
- Drift Monitoring
- Automated Retraining Decision
- Automated Rollback

### Engenharia e automação

- Python
- PySpark
- SQL
- Pandas
- NumPy
- Databricks CLI
- GitHub Actions
- Kaggle API

---

## Dataset

Foi utilizado o dataset público **Credit Card Fraud Detection**, disponibilizado no Kaggle.

```text
mlg-ulb/creditcardfraud
```

O dataset contém transações de cartão de crédito com forte desbalanceamento entre classes legítimas e fraudulentas.

### Download local

```bash
kaggle datasets download \
  -d mlg-ulb/creditcardfraud \
  -p data/raw \
  --unzip

cp data/raw/creditcard.csv data/raw/creditcard_fraud.csv
```

### Upload para o Databricks Volume

```bash
databricks fs cp \
  data/raw/creditcard_fraud.csv \
  dbfs:/Volumes/mlops_dev/fraud/raw/creditcard_fraud.csv \
  --overwrite
```

---

## Estrutura de ambientes

| Ambiente | Catálogo | Schema | Endpoint |
|---|---|---|---|
| DEV | `mlops_dev` | `fraud` | `fraud-detection-endpoint-dev` |
| ACC | `mlops_acc` | `fraud` | `fraud-detection-endpoint-acc` |
| PROD | `mlops_production` | `fraud` | `fraud-detection-endpoint-prod` |

### Criação dos schemas e volumes

```sql
CREATE SCHEMA IF NOT EXISTS mlops_dev.fraud;
CREATE SCHEMA IF NOT EXISTS mlops_acc.fraud;
CREATE SCHEMA IF NOT EXISTS mlops_production.fraud;

CREATE VOLUME IF NOT EXISTS mlops_dev.fraud.raw;
CREATE VOLUME IF NOT EXISTS mlops_acc.fraud.raw;
CREATE VOLUME IF NOT EXISTS mlops_production.fraud.raw;
```

---

## Estrutura do repositório

```text
databricks-banking-fraud-detection/
├── .github/workflows/
│   ├── databricks-bundle-dev.yml
│   ├── databricks-bundle-acc.yml
│   └── databricks-bundle-prod.yml
├── data/raw/
├── docs/
├── notebooks/
│   ├── 01_create_bronze_table.py
│   ├── 02_create_silver_table_v2.py
│   ├── 02_data_quality_checks_v2.py
│   ├── 03_create_gold_table.py
│   ├── 04_create_feature_table.py
│   ├── 04_register_feature_table_uc.py
│   ├── 05_train_models_pipeline.py
│   ├── 06_evaluate_best_model.py
│   ├── 07_promote_model.py
│   ├── 08_batch_inference_v2.py
│   ├── 09_register_serving_model_v2.py
│   ├── 10_monitor_drift.py
│   ├── 10_drift_approval_gate.py
│   ├── 11_compare_champion_challenger.py
│   ├── 12_auto_rollback.py
│   └── 13_retraining_decision_v2.py
├── resources/
│   └── fraud_detection_job.yml
├── scripts/
├── serving/
│   ├── fraud_endpoint_dev.json
│   ├── fraud_endpoint_acc.json
│   ├── fraud_endpoint_prod.json
│   ├── inference_client.py
│   ├── test_payload.json
│   └── results/
├── databricks.yml
├── requirements.txt
└── README.md
```

---

## Pipeline MLOps

O pipeline possui as seguintes tarefas:

| Ordem | Task | Responsabilidade |
|---:|---|---|
| 1 | `create_bronze_table` | Ingestão do CSV e persistência na camada Bronze |
| 2 | `create_silver_table` | Limpeza, tipagem e deduplicação |
| 3 | `data_quality_checks` | Validações e quarentena |
| 4 | `create_gold_table` | Criação de atributos de negócio |
| 5 | `create_feature_table` | Construção do dataset de features |
| 6 | `register_feature_table_uc` | Registro da Feature Table no Unity Catalog |
| 7 | `train_models` | Treinamento de múltiplos modelos |
| 8 | `evaluate_best_model` | Seleção automática do campeão |
| 9 | `monitor_drift` | Cálculo de drift por feature |
| 10 | `retraining_decision` | Decisão de retreinamento |
| 11 | `drift_approval_gate` | Gate de aprovação |
| 12 | `promote_model` | Promoção do modelo candidato |
| 13 | `batch_inference` | Inferência em lote |
| 14 | `register_serving_model` | Registro do wrapper de serving |
| 15 | `compare_champion_challenger` | Comparação entre versões |
| 16 | `auto_rollback` | Rollback automático quando necessário |

---

## Feature Engineering

Além das variáveis PCA `V1` até `V28`, foram criadas features adicionais:

- `transaction_hour`
- `amount_band`
- `log_amount`
- `high_amount_flag`
- `night_transaction_flag`

A Feature Table formal é registrada em:

```text
<catalog>.fraud.fraud_detection_features_uc
```

---

## Modelos treinados

O pipeline avalia os seguintes algoritmos:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Como o problema é fortemente desbalanceado, a seleção prioriza:

1. F1 Score
2. Recall
3. ROC AUC

O modelo selecionado em DEV e ACC foi o `RandomForest`.

---

## Resultados

```text
Best model: RandomForest
F1 Score:   0.8242424242424242
Recall:     0.7157894736842105
```

### Resultado da inferência online

Payload de uma transação fraudulenta conhecida:

```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability_fraud": 0.7916666666666666,
      "risk_level": "high"
    }
  ]
}
```

---

## Data Quality

O pipeline executa validações antes do treinamento:

- campos obrigatórios;
- tipos numéricos;
- valores inválidos;
- duplicidades;
- domínio da classe;
- consistência das features;
- registros destinados à quarentena.

Resultado atual:

```text
DATA_QUALITY_PASSED
records=283726
quarantined_records=0
```

---

## Drift Monitoring

O monitoramento de drift avalia as features e classifica cada uma em:

- `OK`
- `WARNING`
- `DRIFT`

Resultado atual:

```text
OK=34
WARNING=0
DRIFT=0
```

Como não houve drift relevante, o pipeline retornou:

```text
RETRAINING_NOT_REQUIRED
priority=LOW
```

---

## Batch inference

As predições batch são persistidas em:

```text
<catalog>.fraud.fraud_detection_batch_predictions
```

Exemplo de consulta:

```sql
SELECT
  transaction_id,
  prediction,
  ROUND(probability_fraud, 6) AS probability_fraud,
  risk_level,
  model_type,
  prediction_created_at
FROM mlops_dev.fraud.fraud_detection_batch_predictions
ORDER BY probability_fraud DESC
LIMIT 50;
```

---

## Model Registry e Serving

O modelo de serving é registrado como MLflow PyFunc no Unity Catalog.

### DEV

```text
mlops_dev.fraud.fraud_detection_serving_model
version: 4
endpoint: fraud-detection-endpoint-dev
status: READY
```

### ACC

```text
mlops_acc.fraud.fraud_detection_serving_model
version: 1
endpoint: fraud-detection-endpoint-acc
status: READY
```

### PROD

```text
mlops_production.fraud.fraud_detection_serving_model
version: 2
endpoint: fraud-detection-endpoint-prod
status: READY
deployment: DEPLOYMENT_READY
traffic: 100%
```

O wrapper retorna:

- `prediction`
- `probability_fraud`
- `risk_level`

### Dependências do serving

O ambiente de serving foi reduzido para as dependências realmente necessárias ao modelo campeão:

```text
mlflow
cloudpickle
numpy
pandas
scikit-learn
```

Bibliotecas de treinamento como `xgboost`, `lightgbm` e `catboost` não são incluídas no endpoint quando o campeão é um modelo Scikit-learn.

---

## Teste da API REST

### Token OAuth

```bash
export DATABRICKS_CONFIG_PROFILE="fraud-dev"
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"

export DATABRICKS_TOKEN="$(
  databricks auth token fraud-dev --force-refresh |
  jq -r '.access_token'
)"
```

### Chamada com curl

```bash
export DATABRICKS_ENDPOINT_NAME="fraud-detection-endpoint-prod"

curl -sS -X POST \
  "${DATABRICKS_HOST}/serving-endpoints/${DATABRICKS_ENDPOINT_NAME}/invocations" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data @serving/test_payload.json \
  | jq
```

### Cliente Python

```bash
python serving/inference_client.py \
  --payload serving/test_payload.json
```

---

## Databricks Asset Bundles

### DEV

```bash
databricks bundle validate -t dev --profile fraud-dev

databricks bundle deploy \
  -t dev \
  --profile fraud-dev \
  --force

databricks bundle run \
  fraud_detection_mlops_pipeline \
  -t dev \
  --profile fraud-dev
```

### ACC

```bash
databricks bundle validate -t acc --profile fraud-dev

databricks bundle deploy \
  -t acc \
  --profile fraud-dev \
  --force

databricks bundle run \
  fraud_detection_mlops_pipeline \
  -t acc \
  --profile fraud-dev
```

---

## Troubleshooting resolvido

Durante a construção do projeto foram resolvidos os seguintes problemas:

- autenticação OAuth expirada no Databricks CLI;
- profile incorreto selecionado pelo Asset Bundle;
- ausência de assinatura MLflow no modelo registrado;
- excesso de dependências no ambiente do Model Serving;
- endpoint em `UPDATE_FAILED`;
- limite de endpoints na Free Edition;
- erro `401` ao chamar a API;
- erro `RESOURCE_DOES_NOT_EXIST` antes do endpoint ficar ativo;
- diferenças entre configuração ativa e `pending_config`;
- validação do modelo em notebook antes do serving;
- recriação controlada do endpoint entre DEV e ACC.

---

## CI/CD

O projeto possui workflows separados para DEV, ACC e PROD.

```text
Pull Request
    |
    v
Static checks and tests
    |
    v
Bundle validate
    |
    v
Deploy DEV
    |
    v
Validation gate
    |
    v
Deploy ACC
    |
    v
Homologation gate
    |
    v
Deploy PROD
```

---

## Projeto concluído

O pipeline foi validado de ponta a ponta nos três ambientes:

```text
DEV  -> pipeline, registry, serving e API REST validados
ACC  -> pipeline, registry, serving e API REST validados
PROD -> pipeline, registry, serving e API REST validados
```

O projeto está pronto para apresentação em portfólio técnico, demonstrando implementação prática de Lakehouse, MLOps, governança, promoção entre ambientes e inferência online em produção.

---

## Próximos passos

- adicionar testes automatizados de regressão online;
- adicionar um Golden Dataset para regressão de ML;
- implementar `OPTIMIZE`, `VACUUM` e `ANALYZE TABLE`;
- avaliar Liquid Clustering;
- implementar System Tables para custos e observabilidade;
- criar dashboards de operação e monitoramento;
- adicionar testes unitários e de integração ao CI/CD;
- adicionar este projeto e o projeto de Credit Risk ao portfólio profissional.

---

## Autor

**William Ferreira Leandro**

Data Engineering, MLOps, Databricks, Machine Learning, Elastic Stack e Kafka.

GitHub: [@williamfleandro](https://github.com/williamfleandro)
