# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Register Serving Model v2
# MAGIC
# MAGIC Registers a lightweight MLflow PyFunc serving wrapper in Unity Catalog.
# MAGIC
# MAGIC The wrapper returns:
# MAGIC
# MAGIC - `prediction`
# MAGIC - `probability_fraud`
# MAGIC - `risk_level`
# MAGIC
# MAGIC The serving environment contains only the libraries required by the
# MAGIC selected Scikit-learn model.

# COMMAND ----------

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
import sklearn

from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "fraud")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

FEATURE_TABLE = (
    f"{CATALOG_NAME}.{SCHEMA_NAME}."
    "fraud_detection_features_uc"
)

BEST_MODEL_TABLE = (
    f"{CATALOG_NAME}.{SCHEMA_NAME}."
    "fraud_detection_best_model"
)

SERVING_MODEL_NAME = (
    f"{CATALOG_NAME}.{SCHEMA_NAME}."
    "fraud_detection_serving_model"
)

EXPERIMENT_NAME = (
    f"/Shared/fraud-serving-wrapper-"
    f"{CATALOG_NAME}-{SCHEMA_NAME}"
)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 100)
print("SERVING MODEL CONFIGURATION")
print("=" * 100)
print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Feature table: {FEATURE_TABLE}")
print(f"Best model table: {BEST_MODEL_TABLE}")
print(f"Serving model: {SERVING_MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# Read the most recently selected best model.
best_model_rows = (
    spark.table(BEST_MODEL_TABLE)
    .orderBy("created_at", ascending=False)
    .limit(1)
    .collect()
)

if not best_model_rows:
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        f"no record was found in {BEST_MODEL_TABLE}."
    )

best = best_model_rows[0].asDict()

run_id = best.get("mlflow_run_id")
model_type = best.get("model_type")

if not run_id:
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        "mlflow_run_id is missing from the best-model table."
    )

model_uri = f"runs:/{run_id}/model"

print("\n" + "=" * 100)
print("SOURCE MODEL")
print("=" * 100)
print(f"Best model type: {model_type}")
print(f"Best model run_id: {run_id}")
print(f"Model URI: {model_uri}")
print(f"Serving model name: {SERVING_MODEL_NAME}")

# COMMAND ----------

class FraudDetectionServingWrapper(PythonModel):
    """
    MLflow PyFunc wrapper used by Databricks Model Serving.

    The embedded artifact is loaded as a native Scikit-learn model.
    The response contains the predicted class, fraud probability and
    a human-readable risk classification.
    """

    def load_context(self, context):
        import mlflow

        model_artifact_path = context.artifacts["model"]

        self.model = mlflow.sklearn.load_model(
            model_artifact_path
        )

        if not hasattr(self.model, "predict"):
            raise RuntimeError(
                "The embedded model does not expose predict()."
            )

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        import numpy as np
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        predictions = self.model.predict(model_input)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                model_input
            )[:, 1]
        else:
            probabilities = np.asarray(
                predictions,
                dtype=float,
            )

        result = pd.DataFrame(
            {
                "prediction": np.asarray(
                    predictions,
                    dtype=int,
                ),
                "probability_fraud": np.asarray(
                    probabilities,
                    dtype=float,
                ),
            }
        )

        result["risk_level"] = pd.cut(
            result["probability_fraud"],
            bins=[
                -0.01,
                0.20,
                0.60,
                1.01,
            ],
            labels=[
                "low",
                "medium",
                "high",
            ],
            include_lowest=True,
        ).astype(str)

        return result

# COMMAND ----------

print("\n" + "=" * 100)
print("PREPARING INPUT EXAMPLE")
print("=" * 100)

feature_pdf = (
    spark.table(FEATURE_TABLE)
    .limit(20)
    .toPandas()
)

drop_cols = [
    "transaction_id",
    "class",
    "feature_created_at",
]

existing_drop_cols = [
    column
    for column in drop_cols
    if column in feature_pdf.columns
]

input_example = (
    feature_pdf
    .drop(columns=existing_drop_cols)
    .head(5)
    .copy()
)

if input_example.empty:
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        "input_example is empty."
    )

print("Columns used by the serving model:")
for column in input_example.columns:
    print(
        f" - {column}: "
        f"{input_example[column].dtype}"
    )

display(input_example)

# COMMAND ----------

print("\n" + "=" * 100)
print("LOADING SOURCE MODEL")
print("=" * 100)

base_model = mlflow.sklearn.load_model(model_uri)

print(f"Loaded model class: {type(base_model)}")
print(
    "Supports predict_proba: "
    f"{hasattr(base_model, 'predict_proba')}"
)

if not hasattr(base_model, "predict"):
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        "the source model does not expose predict()."
    )

# COMMAND ----------

print("\n" + "=" * 100)
print("GENERATING OUTPUT EXAMPLE")
print("=" * 100)

example_predictions = base_model.predict(
    input_example
)

if hasattr(base_model, "predict_proba"):
    example_probabilities = (
        base_model.predict_proba(
            input_example
        )[:, 1]
    )
else:
    example_probabilities = np.asarray(
        example_predictions,
        dtype=float,
    )

output_example = pd.DataFrame(
    {
        "prediction": np.asarray(
            example_predictions,
            dtype=int,
        ),
        "probability_fraud": np.asarray(
            example_probabilities,
            dtype=float,
        ),
    }
)

output_example["risk_level"] = pd.cut(
    output_example["probability_fraud"],
    bins=[
        -0.01,
        0.20,
        0.60,
        1.01,
    ],
    labels=[
        "low",
        "medium",
        "high",
    ],
    include_lowest=True,
).astype(str)

signature = infer_signature(
    input_example,
    output_example,
)

print("Generated model signature:")
print(signature)

display(output_example)

# COMMAND ----------

print("\n" + "=" * 100)
print("SERVING DEPENDENCIES")
print("=" * 100)

serving_pip_requirements = [
    f"mlflow=={mlflow.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
    f"scikit-learn=={sklearn.__version__}",
]

for requirement in serving_pip_requirements:
    print(f" - {requirement}")

print(
    "\nDependencies intentionally excluded from serving:"
)
print(" - databricks-cli")
print(" - databricks-sdk")
print(" - kaggle")
print(" - requests")
print(" - xgboost")
print(" - lightgbm")
print(" - catboost")

# COMMAND ----------

print("\n" + "=" * 100)
print("LOCAL WRAPPER VALIDATION")
print("=" * 100)

local_wrapper = FraudDetectionServingWrapper()

# Simulates the behavior performed by load_context().
local_wrapper.model = base_model

local_predictions = local_wrapper.predict(
    context=None,
    model_input=input_example,
)

expected_output_columns = [
    "prediction",
    "probability_fraud",
    "risk_level",
]

missing_output_columns = [
    column
    for column in expected_output_columns
    if column not in local_predictions.columns
]

if missing_output_columns:
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        "wrapper output is missing columns: "
        f"{missing_output_columns}"
    )

if len(local_predictions) != len(input_example):
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        "the number of predictions differs from "
        "the number of input rows."
    )

if not local_predictions[
    "probability_fraud"
].between(0.0, 1.0).all():
    raise RuntimeError(
        "SERVING_MODEL_REGISTRATION_FAILED: "
        "probability_fraud contains values outside "
        "the interval [0, 1]."
    )

print("Local wrapper validation succeeded.")

display(local_predictions)

# COMMAND ----------

print("\n" + "=" * 100)
print("REGISTERING SERVING MODEL")
print("=" * 100)

with mlflow.start_run(
    run_name="fraud_detection_serving_wrapper"
) as serving_run:

    mlflow.set_tags(
        {
            "project": "databricks-banking-fraud-detection",
            "environment": CATALOG_NAME,
            "schema": SCHEMA_NAME,
            "source_model_type": str(model_type),
            "source_model_run_id": str(run_id),
            "serving_model_name": SERVING_MODEL_NAME,
            "wrapper_type": "mlflow_pyfunc",
            "serving_framework": "databricks_model_serving",
        }
    )

    mlflow.log_params(
        {
            "source_model_type": str(model_type),
            "source_model_run_id": str(run_id),
            "source_model_uri": model_uri,
            "risk_low_max": 0.20,
            "risk_medium_max": 0.60,
            "input_feature_count": len(
                input_example.columns
            ),
        }
    )

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FraudDetectionServingWrapper(),
        artifacts={
            "model": model_uri,
        },
        registered_model_name=SERVING_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=serving_pip_requirements,
    )

    serving_run_id = serving_run.info.run_id

print("\n" + "=" * 100)
print("MODEL REGISTERED SUCCESSFULLY")
print("=" * 100)
print(f"Serving run ID: {serving_run_id}")
print(f"Registered model: {SERVING_MODEL_NAME}")
print(f"Logged model URI: {model_info.model_uri}")
print(f"Source model run ID: {run_id}")
print(f"Source model type: {model_type}")

# COMMAND ----------

result_message = (
    "SERVING_MODEL_REGISTERED: "
    f"model={SERVING_MODEL_NAME}, "
    f"serving_run_id={serving_run_id}, "
    f"source_model_run_id={run_id}, "
    f"source_model_type={model_type}, "
    f"mlflow_version={mlflow.__version__}, "
    f"sklearn_version={sklearn.__version__}"
)

print(result_message)

dbutils.notebook.exit(result_message)