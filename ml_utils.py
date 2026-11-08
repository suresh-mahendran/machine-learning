from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TEST_SIZE = 0.2
PASS_THRESHOLD = 10
TARGET_COLUMN = "failed"
EXCLUDED_GRADE_COLUMNS = ("G1", "G2", "G3")

MODEL_SPECS = {
    "Logistic Regression": {
        "key": "logistic_regression",
        "filename": "logistic_regression.joblib",
    },
    "Decision Tree": {"key": "decision_tree", "filename": "decision_tree.joblib"},
    "K-Nearest Neighbors": {"key": "knn", "filename": "knn.joblib"},
    "Gaussian Naive Bayes": {"key": "gaussian_nb", "filename": "gaussian_nb.joblib"},
    "Support Vector Machine": {"key": "svm", "filename": "svm.joblib"},
}


def make_target(final_grade: pd.Series) -> pd.Series:
    """Encode failure as 1 (G3 < 10) and passing as 0; reject invalid grades."""
    grades = pd.to_numeric(final_grade, errors="coerce")
    if grades.isna().any():
        raise ValueError("G3 must contain numeric values without missing entries.")
    return (grades < PASS_THRESHOLD).astype("int8").rename(TARGET_COLUMN)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Impute and scale numeric columns; impute and one-hot encode categories.

    The returned transformer is fitted inside each model's training pipeline,
    so statistics and categories are learned from training data only.
    """
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def build_classifiers() -> dict[str, Any]:
    """Create the five classifiers used on the same training/test split."""
    return {
        # A linear model of the probability of failure; class weights give
        # the less frequent failure class more influence during training.
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        # Recursively splits features into decision rules; balanced weights
        # compensate for the unequal class frequencies.
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        # Predicts using the majority class among the five nearest training rows.
        # Scaling the numeric features helps make distances comparable.
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        # Applies Bayes' rule assuming conditionally independent features
        # with Gaussian distributions within each class.
        "Gaussian Naive Bayes": GaussianNB(),
        # Uses an RBF-kernel decision boundary and balanced class weights.
        # Probability estimates are enabled to calculate ROC AUC.
        "Support Vector Machine": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    positive_scores: np.ndarray,
) -> dict[str, float | None]:
    """Evaluate predictions, treating failure (1) as the positive class.

    Accuracy is the fraction of correct predictions; precision is the fraction
    of predicted failures that are actual failures; recall is the fraction of
    actual failures detected. F1 is their harmonic mean. MCC accounts for all
    four confusion-matrix counts. ROC AUC measures ranking using scores, not
    predicted labels, and is undefined when the test set has only one class.
    """
    auc: float | None = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, positive_scores))

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def validate_test_dataframe(
    dataframe: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Check required columns and binary labels, then separate inputs and target."""
    required_columns = [*feature_columns, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataframe]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )
    if dataframe.empty:
        raise ValueError("The uploaded CSV has no rows.")

    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0 (pass) and 1 (fail)."
        )

    features = dataframe.loc[:, feature_columns]
    return features, target.astype("int8")


def positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return scores for failure (class 1) for use in ROC AUC calculation."""
    class_index = list(model.classes_).index(1)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, class_index]
    return model.decision_function(features)
