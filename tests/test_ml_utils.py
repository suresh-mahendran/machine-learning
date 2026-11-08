import numpy as np
import pandas as pd
import unittest

from ml_utils import (
    TARGET_COLUMN,
    calculate_metrics,
    make_target,
    validate_test_dataframe,
)


class MLUtilsTests(unittest.TestCase):
    def test_make_target_uses_fail_as_positive_class(self):
        result = make_target(pd.Series([0, 9, 10, 20]))

        self.assertEqual(result.tolist(), [1, 1, 0, 0])
        self.assertEqual(result.name, TARGET_COLUMN)

    def test_make_target_rejects_non_numeric_grade(self):
        with self.assertRaisesRegex(ValueError, "numeric values"):
            make_target(pd.Series([10, "unknown"]))

    def test_validation_returns_only_required_features_and_binary_target(self):
        frame = pd.DataFrame(
            {"age": [16, 17], "failed": [0, 1], "extra": ["x", "y"]}
        )

        features, target = validate_test_dataframe(frame, ["age"])

        self.assertEqual(features.columns.tolist(), ["age"])
        self.assertEqual(target.tolist(), [0, 1])

    def test_validation_reports_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "age"):
            validate_test_dataframe(pd.DataFrame({"failed": [0]}), ["age"])

    def test_validation_rejects_non_binary_target(self):
        with self.assertRaisesRegex(ValueError, "only 0"):
            validate_test_dataframe(
                pd.DataFrame({"age": [16], "failed": [2]}), ["age"]
            )

    def test_metrics_use_fail_as_positive_class_and_undefined_auc_is_none(self):
        metrics = calculate_metrics(
            np.array([0, 1, 1]),
            np.array([0, 0, 1]),
            np.array([0.1, 0.4, 0.9]),
        )

        self.assertAlmostEqual(metrics["Recall"], 0.5)
        self.assertAlmostEqual(metrics["AUC"], 1.0)
        self.assertGreater(metrics["MCC"], 0)

        single_class_metrics = calculate_metrics(
            np.array([0, 0]), np.array([0, 0]), np.array([0.1, 0.2])
        )
        self.assertIsNone(single_class_metrics["AUC"])


if __name__ == "__main__":
    unittest.main()
