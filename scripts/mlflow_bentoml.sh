#!/usr/bin/env bash

CMD="docker compose exec mlflow python -m mlflow_bentoml"

${CMD}.3_mlflow_log_model
${CMD}.4_mlflow_bentoml_load_model

docker compose exec mlflow bentoml serve --host 0.0.0.0 -p 3000 mlflow_bentoml.5_mlflow_bentoml_service:svc
