# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Register Serving Model v2
# MAGIC
# MAGIC Registers a pyfunc serving wrapper with MLflow signature for Unity Catalog.

# COMMAND ----------

# MAGIC %pip install -q xgboost lightgbm catboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd

from mlflow.pyfunc import PythonModel
from mlflow.models import infer_signature

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "fraud")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.fraud_detection_features_uc"
BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.fraud_detection_best_model"
SERVING_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.fraud_detection_serving_model"
EXPERIMENT_NAME = f"/Shared/fraud-serving-wrapper-{CATALOG_NAME}-{SCHEMA_NAME}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

best = spark.table(BEST_MODEL_TABLE).limit(1).collect()[0].asDict()

run_id = best["mlflow_run_id"]
model_type = best["model_type"]
model_uri = f"runs:/{run_id}/model"

print(f"Best model type: {model_type}")
print(f"Best model run_id: {run_id}")
print(f"Model URI: {model_uri}")
print(f"Serving model name: {SERVING_MODEL_NAME}")

# COMMAND ----------

class FraudDetectionServingWrapper(PythonModel):
    def load_context(self, context):
        import mlflow

        self.model = mlflow.sklearn.load_model(context.artifacts["model"])

    def predict(self, context, model_input):
        import pandas as pd

        predictions = self.model.predict(model_input)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(model_input)[:, 1]
        else:
            probabilities = predictions

        result = pd.DataFrame(
            {
                "prediction": predictions.astype(int),
                "probability_fraud": probabilities.astype(float),
            }
        )

        result["risk_level"] = pd.cut(
            result["probability_fraud"],
            bins=[-0.01, 0.20, 0.60, 1.01],
            labels=["low", "medium", "high"],
        ).astype(str)

        return result

# COMMAND ----------

feature_pdf = spark.table(FEATURE_TABLE).limit(20).toPandas()

drop_cols = ["transaction_id", "class", "feature_created_at"]
input_example = feature_pdf.drop(
    columns=[c for c in drop_cols if c in feature_pdf.columns]
).head(5)

if input_example.empty:
    raise Exception("SERVING_MODEL_REGISTRATION_FAILED: input_example is empty.")

# Load the trained sklearn pipeline only to infer output schema.
base_model = mlflow.sklearn.load_model(model_uri)

example_predictions = base_model.predict(input_example)

if hasattr(base_model, "predict_proba"):
    example_probabilities = base_model.predict_proba(input_example)[:, 1]
else:
    example_probabilities = example_predictions

output_example = pd.DataFrame(
    {
        "prediction": example_predictions.astype(int),
        "probability_fraud": example_probabilities.astype(float),
    }
)

output_example["risk_level"] = pd.cut(
    output_example["probability_fraud"],
    bins=[-0.01, 0.20, 0.60, 1.01],
    labels=["low", "medium", "high"],
).astype(str)

signature = infer_signature(input_example, output_example)

display(input_example)
display(output_example)

# COMMAND ----------

with mlflow.start_run(run_name="fraud_detection_serving_wrapper") as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FraudDetectionServingWrapper(),
        artifacts={"model": model_uri},
        registered_model_name=SERVING_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "pandas",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
    )

    serving_run_id = run.info.run_id

dbutils.notebook.exit(
    f"SERVING_MODEL_REGISTERED: model={SERVING_MODEL_NAME}, "
    f"serving_run_id={serving_run_id}, source_model_run_id={run_id}"
)
