import bentoml
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from sklearn import svm
from sklearn import datasets
import mlflow
from mlflow.models import infer_signature

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

    # Train the model
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    signature = infer_signature(
        iris.data, 
        iris.target
    )
    
    model_name = "iris_clf"
    run_id = mlflow.active_run().info.run_id
    logged_model = mlflow.sklearn.log_model(
        clf, artifact_path = model_name, signature = signature
    )
    model_uri = f"runs:/{run_id}/{model_name}" 

    print(model_uri)
    print(logged_model.model_uri)
    
    mlflow.register_model(model_uri, model_name)
    bento_model = bentoml.mlflow.import_model(
        'iris_clf', 
        logged_model.model_uri,
        labels=mlflow.active_run().data.tags,
        metadata={
        "metrics": mlflow.active_run().data.metrics,
        "params": mlflow.active_run().data.params,
    })

    mlflow.end_run()

if __name__ == "__main__":
    main()