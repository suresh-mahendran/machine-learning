from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import (
    MODEL_SPECS,
    TARGET_COLUMN,
    calculate_metrics,
    positive_class_scores,
    validate_test_dataframe,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def read_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Trained models are missing. Run `python train_models.py` first."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not read the CSV: {error}") from error


def evaluate_all_models(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, dict[str, tuple[object, object]]]:
    rows = {}
    predictions_by_model = {}
    for model_name, specification in MODEL_SPECS.items():
        model_path = MODEL_DIR / specification["filename"]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file is missing: {model_path.name}. "
                "Run `python train_models.py` to generate it."
            )
        model = load_model(str(model_path))
        predictions = model.predict(features)
        scores = positive_class_scores(model, features)
        rows[model_name] = calculate_metrics(target, predictions, scores)
        predictions_by_model[model_name] = (model, predictions)

    metrics = pd.DataFrame.from_dict(rows, orient="index")[METRIC_COLUMNS]
    return metrics, predictions_by_model


def main() -> None:
    st.set_page_config(page_title="Student Performance Classifiers", layout="wide")
    st.title("Student Performance: Pass/Fail Classifiers")
    st.write(
        "Compare five saved classifiers on labeled test data. "
        "The app loads trained pipelines and does not retrain on uploaded data."
    )
    st.caption(
        "Target convention: `failed = 1` when G3 < 10; `failed = 0` otherwise. "
        "Earlier grade columns G1/G2 and final grade G3 are excluded from inputs."
    )

    try:
        metadata = read_metadata()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        st.error(str(error))
        st.stop()

    data_choice = st.radio(
        "Choose test data",
        ["Bundled held-out test set", "Upload a labeled CSV"],
        horizontal=True,
    )
    dataframe: pd.DataFrame | None = None
    if data_choice == "Bundled held-out test set":
        test_data_path = PROJECT_DIR / "test_data.csv"
        if not test_data_path.is_file():
            st.error("Bundled test_data.csv is missing.")
            st.stop()
        dataframe = pd.read_csv(test_data_path)
    else:
        upload = st.file_uploader(
            "Upload a CSV with all required features and a `failed` target column",
            type=["csv"],
        )
        if upload is None:
            st.info("Upload a test CSV to see model results.")
            st.stop()
        try:
            dataframe = read_csv(upload)
        except ValueError as error:
            st.error(str(error))
            st.stop()

    try:
        features, target = validate_test_dataframe(
            dataframe, metadata["feature_columns"]
        )
        comparison, predictions_by_model = evaluate_all_models(features, target)
    except (ValueError, KeyError, FileNotFoundError) as error:
        st.error(str(error))
        st.stop()

    if comparison["AUC"].isna().any():
        st.warning(
            "AUC is unavailable for this test CSV because it does not contain both "
            "target classes."
        )
    st.subheader(f"Evaluation on {len(dataframe)} labeled rows")
    st.dataframe(comparison.style.format("{:.3f}", na_rep="N/A"), use_container_width=True)
    st.caption(
        "Precision, recall, and F1 use the fail class (`failed = 1`) as positive. "
        "AUC uses each model's score for that class."
    )

    selected_model = st.selectbox("Model details", list(MODEL_SPECS))
    model, predictions = predictions_by_model[selected_model]
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    st.subheader(f"{selected_model}: confusion matrix")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual pass (0)", "Actual fail (1)"],
            columns=["Predicted pass (0)", "Predicted fail (1)"],
        ),
        use_container_width=True,
    )

    st.subheader("Classification report")
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["Pass", "Fail"],
        zero_division=0,
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)


if __name__ == "__main__":
    main()
