"""
churn_script_logging_and_tests.py

Unit tests and logging for churn_library.py.
Runs all pipeline stages and logs pass/fail to churn_library.log.

Author: Eduardo Nicacio
Date:   2026-11-16
"""

# pylint: disable=broad-except

import logging
import os

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)
        logging.info("test_import: SUCCESS - file found")
    except FileNotFoundError as err:
        logging.error("test_import: FAILED - file not found: %s", err)
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "test_import: SUCCESS - DataFrame shape %s", df.shape
        )
    except AssertionError as err:
        logging.error("test_import: FAILED - DataFrame is empty")
        raise err


def test_eda(perform_eda):
    """Test EDA."""
    expected_files = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "total_transaction_distribution.png",
        "heatmap.png",
    ]
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)
        for fname in expected_files:
            fpath = os.path.join("./images/eda", fname)
            assert os.path.isfile(fpath), f"Missing EDA file: {fpath}"
        logging.info("test_eda: SUCCESS - all EDA files created")
    except Exception as err:
        logging.error("test_eda: FAILED - %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        df = encoder_helper(df, cls.CATEGORY_COLUMNS, "Churn")
        for col in cls.CATEGORY_COLUMNS:
            encoded_col = f"{col}_Churn"
            assert encoded_col in df.columns, (
                f"Missing encoded column: {encoded_col}"
            )
        logging.info(
            "test_encoder_helper: SUCCESS - encoded columns present"
        )
    except Exception as err:
        logging.error("test_encoder_helper: FAILED - %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        df = cls.encoder_helper(df, cls.CATEGORY_COLUMNS, "Churn")
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, "Churn"
        )
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info(
            "test_perform_feature_engineering: SUCCESS - "
            "train=%d test=%d",
            x_train.shape[0],
            x_test.shape[0],
        )
    except Exception as err:
        logging.error(
            "test_perform_feature_engineering: FAILED - %s", err
        )
        raise err


def test_train_models(train_models):
    """Test model training."""
    expected_models = [
        os.path.join("./models", "rfc_model.pkl"),
        os.path.join("./models", "logistic_model.pkl"),
    ]
    expected_images = [
        os.path.join("./images/results", "logistic_results.png"),
        os.path.join("./images/results", "rf_results.png"),
        os.path.join("./images/results", "roc_curve_result.png"),
        os.path.join("./images/results", "feature_importances.png"),
    ]
    try:
        df = cls.import_data(DATA_PATH)
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        df = cls.encoder_helper(df, cls.CATEGORY_COLUMNS, "Churn")
        x_train, x_test, y_train, y_test = (
            cls.perform_feature_engineering(df, "Churn")
        )
        train_models(x_train, x_test, y_train, y_test)

        for path in expected_models + expected_images:
            assert os.path.isfile(path), f"Missing output: {path}"
        logging.info(
            "test_train_models: SUCCESS - all outputs found"
        )
    except Exception as err:
        logging.error("test_train_models: FAILED - %s", err)
        raise err


if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("All tests completed. Check logs/churn_library.log for details.")
