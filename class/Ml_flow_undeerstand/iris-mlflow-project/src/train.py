import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import pickle
import os

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_random_forest(X_train, y_train, X_test, y_test, **params):
    """
    Trains a RF in a nested run, logs params/metrics/model to that nested run
    and returns (accuracy, trained_model, nested_run_id).
    """
    with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}", nested=True) as run:
        # log params individually (safer)
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # log model artifact explicitly to the nested run
        mlflow.sklearn.log_model(model, "model")

        nested_run_id = run.info.run_id
        print(f"Nested run complete: run_id={nested_run_id}, params={params}, acc={acc:.4f}")
        return acc, model, nested_run_id

def hyperparameter_tuning_nested():
    mlflow.set_experiment("Iris Classification Hyperparameter")

    param_grid = [
        {"n_estimators": 10, "max_depth": 3},
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": None},
    ]

    best_acc = -1
    best_params = None
    best_model = None
    best_nested_run_id = None

    # parent run
    with mlflow.start_run(run_name="Iris_Experiment") as parent_run:
        for params in param_grid:
            acc, model, nested_run_id = train_random_forest(X_train, y_train, X_test, y_test, **params)

            # if this is the best so far, save model in parent run (and keep reference)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_model = model
                best_nested_run_id = nested_run_id

                # Log the best model under the parent run as "best_model"
                # This keeps a single canonical "best" model in the parent run's artifacts
                mlflow.sklearn.log_model(best_model, artifact_path="best_model")

                # Also save a local pickle and log it as an artifact (optional)
                local_path = "best_model.pkl"
                with open(local_path, "wb") as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact(local_path, artifact_path="best_model_files")
                # cleanup local file
                try:
                    os.remove(local_path)
                except OSError:
                    pass

                # If you want to reference the nested run that produced this model:
                mlflow.log_param("best_nested_run_id", best_nested_run_id)

        # log best params (as individual params)
        if best_params:
            mlflow.log_params({
                "best_n_estimators": best_params["n_estimators"],
                "best_max_depth": str(best_params["max_depth"])
            })
        mlflow.log_metric("best_accuracy", best_acc)
        # Or store whole dict as JSON artifact:
        mlflow.log_text(json.dumps(best_params), "best_params.json")

        print(f"Parent run complete. Best params: {best_params}, best_acc={best_acc:.4f}, best_nested_run_id={best_nested_run_id}")

        # OPTIONAL: register the best model in Model Registry (requires an MLflow tracking server + registry)
        # If you want to register the model, uncomment and adapt the two lines below.
        # The source uses the parent run artifact path we logged above:
        model_uri = f"runs:/{parent_run.info.run_id}/best_model"
        mlflow.register_model(model_uri, name="IrisRFBest")

if __name__ == "__main__":
    hyperparameter_tuning_nested()
