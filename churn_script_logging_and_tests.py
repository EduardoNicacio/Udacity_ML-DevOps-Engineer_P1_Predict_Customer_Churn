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

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"
EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _load_encoded_split():
    """
    Load the dataset and return the leak-free train/test split.

    input:
            None
    output:
            x_train, x_test, y_train, y_test
    """
    df = cls.import_data(DATA_PATH)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return cls.perform_feature_engineering(df, "Churn")


def test_import(import_data):
    """
    Verify that import_data loads a non-empty DataFrame from disk.

    input:
            import_data: the churn_library.import_data function
    output:
            None
    """
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


def test_create_output_directories(create_output_directories):
    """
    Verify that create_output_directories creates the folders used by
    the project for EDA images, result images and saved models.

    input:
            create_output_directories: the churn_library function of
                the same name
    output:
            None
    """
    try:
        create_output_directories()
        for directory in (EDA_DIR, RESULTS_DIR, MODELS_DIR):
            assert os.path.isdir(directory), (
                f"Missing directory: {directory}"
            )
        logging.info(
            "test_create_output_directories: SUCCESS - "
            "all directories exist"
        )
    except Exception as err:
        logging.error(
            "test_create_output_directories: FAILED - %s", err
        )
        raise err


def test_eda(perform_eda):
    """
    Verify that perform_eda saves the five expected EDA images to the
    EDA directory.

    input:
            perform_eda: the churn_library.perform_eda function
    output:
            None
    """
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
            fpath = os.path.join(EDA_DIR, fname)
            assert os.path.isfile(fpath), f"Missing EDA file: {fpath}"
        logging.info("test_eda: SUCCESS - all EDA files created")
    except Exception as err:
        logging.error("test_eda: FAILED - %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    """
    Verify that encoder_helper adds one mean-response encoded column
    per categorical column and that the encoded values are non-empty.

    input:
            encoder_helper: the churn_library.encoder_helper function
    output:
            None
    """
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
            assert df[encoded_col].notna().all(), (
                f"Encoded column has empty values: {encoded_col}"
            )
        logging.info(
            "test_encoder_helper: SUCCESS - encoded columns present"
        )
    except Exception as err:
        logging.error("test_encoder_helper: FAILED - %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Verify that perform_feature_engineering returns four non-empty
    train/test arrays with matching sample counts.

    input:
            perform_feature_engineering: the churn_library function of
                the same name
    output:
            None
    """
    try:
        df = cls.import_data(DATA_PATH)
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, "Churn"
        )
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert x_train.shape[0] == len(y_train)
        assert x_test.shape[0] == len(y_test)
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


def test_classification_report_image(classification_report_image):
    """
    Verify that classification_report_image saves the logistic
    regression and random forest report images.

    input:
            classification_report_image: the churn_library function of
                the same name
    output:
            None
    """
    expected_files = [
        os.path.join(RESULTS_DIR, "logistic_results.png"),
        os.path.join(RESULTS_DIR, "rf_results.png"),
    ]
    try:
        _, _, y_train, y_test = _load_encoded_split()
        rng = np.random.default_rng(42)
        y_train_preds = rng.choice(
            np.unique(y_train), size=len(y_train)
        )
        y_test_preds = rng.choice(np.unique(y_test), size=len(y_test))
        classification_report_image(
            y_train,
            y_test,
            y_train_preds,
            y_train_preds,
            y_test_preds,
            y_test_preds,
        )
        for fpath in expected_files:
            assert os.path.isfile(fpath), f"Missing output: {fpath}"
        logging.info(
            "test_classification_report_image: SUCCESS - "
            "report images created"
        )
    except Exception as err:
        logging.error(
            "test_classification_report_image: FAILED - %s", err
        )
        raise err


def test_feature_importance_plot(feature_importance_plot):
    """
    Verify that feature_importance_plot saves the SHAP summary and
    the random forest feature importance images.

    input:
            feature_importance_plot: the churn_library function of the
                same name
    output:
            None
    """
    expected_files = [
        os.path.join(RESULTS_DIR, "shap_summary.png"),
        os.path.join(RESULTS_DIR, "feature_importances.png"),
    ]
    try:
        x_train, x_test, y_train, _ = _load_encoded_split()
        model = RandomForestClassifier(
            n_estimators=10, max_depth=4, random_state=42
        )
        model.fit(x_train, y_train)
        feature_importance_plot(
            model,
            x_test,
            os.path.join(RESULTS_DIR, "shap_summary.png"),
        )
        for fpath in expected_files:
            assert os.path.isfile(fpath), f"Missing output: {fpath}"
        logging.info(
            "test_feature_importance_plot: SUCCESS - "
            "importance images created"
        )
    except Exception as err:
        logging.error(
            "test_feature_importance_plot: FAILED - %s", err
        )
        raise err


def test_train_models(train_models):
    """
    Verify that train_models saves both model files and all result
    images.

    input:
            train_models: the churn_library.train_models function
    output:
            None
    """
    expected_models = [
        os.path.join(MODELS_DIR, "rfc_model.pkl"),
        os.path.join(MODELS_DIR, "logistic_model.pkl"),
    ]
    expected_images = [
        os.path.join(RESULTS_DIR, "logistic_results.png"),
        os.path.join(RESULTS_DIR, "rf_results.png"),
        os.path.join(RESULTS_DIR, "roc_curve_result.png"),
        os.path.join(RESULTS_DIR, "feature_importances.png"),
    ]
    try:
        x_train, x_test, y_train, y_test = _load_encoded_split()
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
    test_create_output_directories(cls.create_output_directories)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_classification_report_image(cls.classification_report_image)
    test_feature_importance_plot(cls.feature_importance_plot)
    test_train_models(cls.train_models)

    print("All tests completed. Check logs/churn_library.log for details.")
