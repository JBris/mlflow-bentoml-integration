import bentoml
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from sklearn import svm
from sklearn import datasets
import mlflow
from mlflow.models import infer_signature
import tempfile
import shutil
import subprocess
import os
from os.path import join
import boto3
import glob

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    print(config)
    EXPERIMENT_CONFIG = instantiate(config["experiment"])

    mlflow.set_tracking_uri(EXPERIMENT_CONFIG.tracking_uri)
    experiment_name = EXPERIMENT_CONFIG.name

    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    mlflow.set_tag("task", "bento_ml_test")

    # Load training data set
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    model_uri_prefix = "models:/"
    model_uri_suffix = "/latest"
    model_name = "iris_clf"
    model_uri = f"{model_uri_prefix}{model_name}{model_uri_suffix}"

    loaded_model = mlflow.sklearn.load_model(
        model_uri = model_uri
    )
    print("Loaded MLFlow model")
    print(loaded_model)

    saved_model = bentoml.sklearn.save_model("iris_clf_sklearn", loaded_model)
    print("Saved MLFlow model")
    print(saved_model)

    model_version = "1.0"
    # The Poetry .venv directory causes bentoml build to hang (due to large directory size)
    # So we just copy the build dependencies elsewhere
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as f:
        shutil.copy(join("/workspace", "models", "bentofile_mlflow.yaml"), f)
        shutil.copytree(join("/workspace", "mlflow_bentoml"), join(f, "mlflow_bentoml"))

        subprocess.check_call(
            ["bentoml", "build", "--version", model_version, "-f", "bentofile_mlflow.yaml"], 
            cwd=f
        )
    
    model_dir = join(os.environ["BENTOML_HOME"], "bentos", "iris_classifier", model_version)

    s3_client = boto3.client(
        "s3",
        verify=False,
        endpoint_url = os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )

    for f in glob.iglob(
        "**/*", recursive = True, 
        root_dir= model_dir
    ): 
        full_path = join(model_dir, f)
        if os.path.isdir(full_path):
            continue
        
        s3_client.upload_file(
            full_path, "bento", 
            join("bentos", "iris_classifier", model_version, f)
        )
            
    s3_client.close()
    mlflow.end_run()
    
if __name__ == "__main__":
    main()