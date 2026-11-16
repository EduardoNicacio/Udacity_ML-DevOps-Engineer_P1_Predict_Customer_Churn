"""
churn_library.py

Library of functions to find customers who are likely to churn.
Implements data import, EDA, feature engineering, model training,
and evaluation following clean-code and PEP 8 principles.

Author: Eduardo Nicacio
Date:   2026-11-16
"""

# The public signature of classification_report_image is fixed by the
# project specification and therefore exempt from the argument-count check.
# pylint: disable=too-many-positional-arguments

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

matplotlib.use("Agg", force=True)  # non-interactive backend, headless safe

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"

CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    return pd.read_csv(pth)


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    plt.figure(figsize=(10, 5))
    df["Churn"].hist()
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = Existing, 1 = Attrited)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(EDA_DIR, "churn_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    df["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Customer Age")
    plt.ylabel("Count")
    plt.savefig(os.path.join(EDA_DIR, "customer_age_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.ylabel("Proportion")
    plt.savefig(
        os.path.join(EDA_DIR, "marital_status_distribution.png")
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Total Transaction Count Distribution")
    plt.xlabel("Total Transaction Count")
    plt.ylabel("Density")
    plt.savefig(
        os.path.join(EDA_DIR, "total_transaction_distribution.png")
    )
    plt.close()

    plt.figure(figsize=(20, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="Dark2_r", linewidths=2)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(EDA_DIR, "heatmap.png"))
    plt.close("all")


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features with target (mean-response) encoding.

    For every column in category_lst, a mapping from each category to
    the mean value of the response is computed on the dataframe passed
    in, and a new column named "{col}_{response}" is added. Pass only
    the training data to this function when preparing model features,
    so the encoding statistics never use rows from the test set.

    input:
            df: pandas dataframe containing the category columns and
                the response column
            category_lst: list of categorical column names to encode
            response: name of the binary response column used to
                compute the per-category mean
    output:
            df: updated dataframe with one "{col}_{response}" column
                added per entry in category_lst
    """
    for col in category_lst:
        df[f"{col}_{response}"] = df[col].map(
            df.groupby(col)[response].mean()
        )
    return df


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets, then encode categories.

    The split is performed first, and the target encoding for every
    categorical column is fitted on the training fold only. Test rows
    whose category was unseen in training receive the training-fold
    mean response. This prevents target leakage from the test set
    into the training features.

    input:
              df: pandas dataframe containing the raw category
                  columns and the response column
              response: name of the binary response column
    output:
              x_train, x_test, y_train, y_test
    """
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42
    )

    global_mean = df_train[response].mean()
    for col in CATEGORY_COLUMNS:
        mapping = df_train.groupby(col)[response].mean()
        for split in (df_train, df_test):
            split[f"{col}_{response}"] = (
                split[col].map(mapping).fillna(global_mean)
            )

    x_train = df_train[keep_cols]
    x_test = df_test[keep_cols]
    y_train = df_train[response]
    y_test = df_test[response]
    return x_train, x_test, y_train, y_test


def _save_report_image(report, pth):
    """
    Render a classification report as a PNG image.

    input:
            report: classification report text
            pth: output image path
    output:
            None
    """
    plt.figure(figsize=(10, 5))
    plt.text(
        0.01,
        0.99,
        report,
        {"family": "monospace", "size": 10},
        va="top",
    )
    plt.axis("off")
    plt.savefig(pth, bbox_inches="tight")
    plt.close()


def classification_report_image(  # pylint: disable=too-many-arguments
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Save classification reports as images.

    Renders the train and test classification reports of both models
    into RESULTS_DIR: logistic_results.png (Logistic Regression) and
    rf_results.png (Random Forest).

    input:
            y_train: true labels of the training set
            y_test: true labels of the test set
            y_train_preds_lr: Logistic Regression predictions on the
                training set
            y_train_preds_rf: Random Forest predictions on the
                training set
            y_test_preds_lr: Logistic Regression predictions on the
                test set
            y_test_preds_rf: Random Forest predictions on the test
                set
    output:
            None
    """
    create_output_directories()

    lr_report = (
        "Logistic Regression - Train\n"
        + classification_report(y_train, y_train_preds_lr)
        + "\nLogistic Regression - Test\n"
        + classification_report(y_test, y_test_preds_lr)
    )
    _save_report_image(
        lr_report, os.path.join(RESULTS_DIR, "logistic_results.png")
    )

    rf_report = (
        "Random Forest - Train\n"
        + classification_report(y_train, y_train_preds_rf)
        + "\nRandom Forest - Test\n"
        + classification_report(y_test, y_test_preds_rf)
    )
    _save_report_image(
        rf_report, os.path.join(RESULTS_DIR, "rf_results.png")
    )


def _as_class_list(shap_values):
    """
    Normalise SHAP values to a list of per-class arrays.

    input:
            shap_values: output of TreeExplainer.shap_values
    output:
            shallow values as a list of arrays
    """
    if isinstance(shap_values, list):
        return shap_values
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [
            shap_values[:, :, idx]
            for idx in range(shap_values.shape[2])
        ]
    return shap_values


def feature_importance_plot(model, x_data, output_pth):
    """
    Save feature importance plot.

    input:
            model, x_data, output path
    output:
            None
    """
    create_output_directories()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(
        _as_class_list(shap_values),
        x_data,
        plot_type="bar",
        show=False,
    )
    # pylint: disable-next=unreachable
    plt.savefig(output_pth, bbox_inches="tight")
    plt.close("all")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[idx] for idx in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(
        os.path.join(RESULTS_DIR, "feature_importances.png"),
        bbox_inches="tight",
    )
    plt.close("all")


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            train/test data
    output:
            None
    """
    create_output_directories()

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["sqrt", "log2"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    cv_rfc = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
    )
    cv_rfc.fit(x_train, y_train)
    rfc_model = cv_rfc.best_estimator_

    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    joblib.dump(rfc_model, os.path.join(MODELS_DIR, "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODELS_DIR, "logistic_model.pkl"))

    plt.figure(figsize=(10, 5))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(
        lrc,
        x_test,
        y_test,
        ax=axis,
        name="Logistic Regression",
    )
    RocCurveDisplay.from_estimator(
        rfc_model,
        x_test,
        y_test,
        ax=axis,
        name="Random Forest",
    )
    plt.title("ROC Curve Comparison")
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_result.png"))
    plt.close()

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    feature_importance_plot(
        rfc_model,
        x_test,
        os.path.join(RESULTS_DIR, "shap_summary.png"),
    )


def main():
    """
    Run the complete churn prediction pipeline end to end.

    input:
            None
    output:
            None
    """
    create_output_directories()

    df = import_data(DATA_PATH)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    perform_eda(df)

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df, "Churn"
    )
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
