from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from ml_utils import (
    EXCLUDED_GRADE_COLUMNS,
    MODEL_SPECS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    build_classifiers,
    build_preprocessor,
    calculate_metrics,
    make_target,
    positive_class_scores,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "student-por.csv"
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"


def train(data_path: Path) -> dict[str, dict[str, float | None]]:
    """Train all classifiers once and evaluate them on the same held-out rows."""
    source = pd.read_csv(data_path, sep=";")
    if "G3" not in source.columns:
        raise ValueError("The source CSV must contain the final-grade column G3.")
    if source["G3"].isna().any():
        raise ValueError("The source CSV contains missing final grades in G3.")

    # Exclude all grade columns so the inputs do not reveal the target or rely
    # on earlier grades, and stratify to preserve the pass/fail proportions.
    target = make_target(source["G3"])
    features = source.drop(
        columns=[*EXCLUDED_GRADE_COLUMNS], errors="ignore"
    )
    feature_columns = features.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    test_data = X_test.reset_index(drop=True).copy()
    test_data[TARGET_COLUMN] = y_test.reset_index(drop=True)
    test_data.to_csv(TEST_DATA_PATH, index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float | None]] = {}
    for model_name, classifier in build_classifiers().items():
        # Fit preprocessing and classifier together on training data only.
        # Save the complete pipeline so evaluation never refits preprocessing.
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = positive_class_scores(pipeline, X_test)
        results[model_name] = calculate_metrics(y_test, predictions, scores)
        model_path = MODEL_DIR / MODEL_SPECS[model_name]["filename"]
        joblib.dump(pipeline, model_path)

    metadata = {
        "dataset": "UCI Student Performance - Portuguese course",
        "source_file": "data/student-por.csv",
        "source_url": "https://archive.ics.uci.edu/dataset/320/student%2Bperformance",
        "target_column": TARGET_COLUMN,
        "target_definition": "1 if G3 < 10 (fail), otherwise 0 (pass)",
        "positive_class": 1,
        "excluded_grade_columns": list(EXCLUDED_GRADE_COLUMNS),
        "feature_columns": feature_columns,
        "rows": int(len(source)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_class_counts": {
            "pass": int((y_test == 0).sum()),
            "fail": int((y_test == 1).sum()),
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "models": MODEL_SPECS,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the five student-failure classifiers."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the UCI student-por.csv source file.",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset file not found: {args.data}")

    results = train(args.data)
    print(f"Saved held-out test data to {TEST_DATA_PATH}")
    print(f"Saved five trained model pipelines and metadata to {MODEL_DIR}")
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
