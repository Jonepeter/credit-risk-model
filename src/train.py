import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    logger.info(f"Data split into train ({len(X_train)}) and test ({len(X_test)}) sets")
    return X_train, X_test, y_train, y_test

def get_model_config() -> Dict[str, Any]:
    """Return configuration of models and hyperparameters for tuning."""
    return {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"]
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(class_weight="balanced"),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0]
            }
        },
        "xgboost": {
            "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        }
    }

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, float]:
    """Evaluate model performance and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    logger.info(f"\nEvaluation metrics for {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return metrics

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    search_method: str = "random",
    n_iter: int = 10
) -> Dict[str, Any]:
    """Train and evaluate multiple models with hyperparameter tuning."""
    mlflow.set_experiment("Credit_Risk_Modeling")
    model_config = get_model_config()
    best_model = None
    best_score = 0
    results = {}
    
    for model_name, config in model_config.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            # Log model info
            mlflow.log_param("model_type", model_name)
            
            # Hyperparameter tuning
            if search_method == "grid":
                search = GridSearchCV(
                    config["model"],
                    config["params"],
                    cv=5,
                    scoring="roc_auc",
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    config["model"],
                    config["params"],
                    n_iter=n_iter,
                    cv=5,
                    scoring="roc_auc",
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            search.fit(X_train, y_train)
            model = search.best_estimator_
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test, model_name)
            
            # Log metrics and parameters
            mlflow.log_metrics(metrics)
            mlflow.log_params(search.best_params_)
            mlflow.sklearn.log_model(model, model_name)
            
            # Track best model
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model
                mlflow.set_tag("best_model", "True")
            else:
                mlflow.set_tag("best_model", "False")
            
            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "params": search.best_params_
            }
    
    # Register best model
    if best_model:
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="CreditRiskModel"
        )
        logger.info(f"Best model registered with ROC-AUC: {best_score:.4f}")
    # Save the best model to a file using joblib
    import joblib
    import os
    os.makedirs("../model", exist_ok=True)
    joblib.dump(best_model, "../model/best_model.pkl")
    logger.info("Best model saved to best_model.pkl")

    # Display 10 most important features of the best model
    if best_model is not None:
        # Try to get feature importances or coefficients
        feature_names = None
        if hasattr(X_train, "columns"):
            feature_names = X_train.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        importances = None
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            # For linear models, coef_ may be 2D (n_classes, n_features)
            coef = best_model.coef_
            if coef.ndim == 1:
                importances = coef
            else:
                # Take the norm across classes
                import numpy as np
                importances = np.linalg.norm(coef, axis=0)
        elif hasattr(best_model, "steps"):
            # Pipeline: try to get last estimator
            last_step = best_model.steps[-1][1]
            if hasattr(last_step, "feature_importances_"):
                importances = last_step.feature_importances_
            elif hasattr(last_step, "coef_"):
                coef = last_step.coef_
                if coef.ndim == 1:
                    importances = coef
                else:
                    import numpy as np
                    importances = np.linalg.norm(coef, axis=0)
        if importances is not None:
            # Get top 10 features
            import numpy as np
            indices = np.argsort(np.abs(importances))[::-1][:10]
            logger.info("Top 10 important features of the best model:")
            for rank, idx in enumerate(indices, 1):
                logger.info(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")
        else:
            logger.info("Best model does not provide feature importances or coefficients.")

    return results
    


