# Databricks notebook source
# MAGIC %md
# MAGIC # 10 Monitor Drift
# MAGIC
# MAGIC Projeto: Databricks Banking — Fraud Detection

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

DOMAIN = "fraud_detection"
TARGET_COLUMN = "Class"
RAW_FILE = "creditcard.csv"

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Domain: {DOMAIN}")
print(f"Target: {TARGET_COLUMN}")
print("TODO: implementar este notebook seguindo o padrão do projeto Databricks MLOps Churn Lab.")
