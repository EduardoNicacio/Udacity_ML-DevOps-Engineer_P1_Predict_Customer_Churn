# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)

---

## Project Description

> This project identifies customers of a credit card company who are likely to
> close ("churn") their account, so the business can target them with retention
> campaigns before they leave.
>
> The pipeline loads the bank customer dataset, creates a binary churn target
> from the `Attrition_Flag` column, performs exploratory data analysis, encodes
> categorical features using their mean churn rate, and splits the data into
> training (70%) and test (30%) sets.
>
> Two classification models are trained and compared: a **Random Forest**
> tuned with `GridSearchCV` and a **Logistic Regression**. Their performance is
> evaluated with classification reports and ROC curves, and the Random Forest
> is further interpreted with SHAP values and feature importances in order to
> understand which behaviors drive churn.

---

## Files and Data Description

The repository is organised around a single reusable library module, a test /
logging script, and a reference notebook that documents the analysis.

### Main Files

- `churn_library.py`
  > Contains the whole machine learning pipeline as small, documented, reusable
  > functions: `import_data`, `perform_eda`, `encoder_helper`,
  > `perform_feature_engineering`, `classification_report_image`,
  > `feature_importance_plot` and `train_models`. It also defines the output
  > directory constants (`EDA_DIR`, `RESULTS_DIR`, `MODELS_DIR`, `DATA_PATH`)
  > and creates the directories on demand. Running the file executes the full
  > end-to-end pipeline.

- `churn_script_logging_and_tests.py`
  > Contains the unit tests for each library stage (`test_import`, `test_eda`,
  > `test_encoder_helper`, `test_perform_feature_engineering`,
  > `test_train_models`). Every test wraps its assertions in `try/except`
  > blocks and records a timestamped `SUCCESS` / `FAILED` entry in
  > `logs/churn_library.log`, so the file doubles as an execution audit trail.

- `churn_notebook.ipynb`
  > The original exploratory notebook supplied with the project. It documents
  > the analysis that the library reproduces (EDA plots, encoding, modelling and
  > SHAP interpretation) and therefore serves as the reference implementation
  > and behavioural specification. It is left unmodified.

---

### Data

- `data/bank_data.csv`
  > The Kaggle *Credit Card Customers* dataset. Each row is one credit card
  > customer, described by demographic features (`Customer_Age`, `Gender`,
  > `Dependent_count`, `Education_Level`, `Marital_Status`,
  > `Income_Category`), account features (`Card_Category`, `Months_on_book`,
  > `Total_Relationship_Count`, `Months_Inactive_12_mon`,
  > `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`,
  > `Avg_Open_To_Buy`), transaction behaviour (`Total_Amt_Chng_Q4_Q1`,
  > `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`,
  > `Avg_Utilization_Ratio`) and the target column `Attrition_Flag`
  > (`Existing Customer` / `Attrited Customer`).
  >
  > The binary target `Churn` is derived from `Attrition_Flag`
  > (1 = attrited, 0 = existing). Categorical columns are encoded as the mean
  > churn rate per category (`<column>_Churn`), which keeps the feature space
  > numeric and directly interpretable.

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

---

## Environment Setup

> Before running the pipeline or the tests, create the virtual environment and
> install the dependencies (Windows, PowerShell):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> The environment used for this submission runs Python 3.14; `shap` is pinned
> as `shap>=0.54.1` because version 0.50.0 provides no Python 3.14 wheels.
> The pipeline is CPU-bound (scikit-learn models only), so no CUDA build is
> required.

---

## Running the Files

### 1. Run the Pipeline

```bash
python churn_library.py
```

> Running this script executes the complete pipeline: it loads
> `data/bank_data.csv`, derives the `Churn` target, generates and saves the
> five EDA plots, encodes the five categorical features, performs the
> train/test split (70/30, `random_state=42`), tunes a Random Forest with a
> 5-fold `GridSearchCV`, trains the Logistic Regression, saves both fitted
> models with `joblib`, and writes the ROC curve comparison, classification
> reports and feature importance plots. The grid search trains 120 model
> combinations, so the run typically takes a few minutes.

---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py
```

> The test script re-runs each pipeline stage and asserts the expected
> side effects: the CSV loads into a non-empty DataFrame, all five EDA images
> are created, every `<category>_Churn` column is present, the four train/test
> arrays are non-empty, and both model files and the four result images exist.
> Each check appends a timestamped `SUCCESS` or `FAILED` message (including the
> exception text) to `logs/churn_library.log`; any failure is re-raised so the
> process exits with a non-zero status. The log file is truncated at the start
> of every run, so it always reflects the most recent execution.

---

## Expected Outputs

> A successful run produces the following artefacts:

- Models:
  - `models/rfc_model.pkl` — grid-searched Random Forest classifier
  - `models/logistic_model.pkl` — logistic regression classifier

- Images:
  - EDA plots in `images/eda/`
    - `churn_distribution.png` — histogram of the binary churn target
    - `customer_age_distribution.png` — histogram of customer age
    - `marital_status_distribution.png` — normalized marital status counts
    - `total_transaction_distribution.png` — transaction count distribution
      with KDE
    - `heatmap.png` — correlation heatmap of all numeric features
  - Model evaluation plots in `images/results/`
    - `logistic_results.png` — logistic regression classification reports
      (train and test)
    - `rf_results.png` — random forest classification reports (train and test)
    - `roc_curve_result.png` — ROC curves for both models on one axes
    - `feature_importances.png` — random forest feature importances
    - `shap_summary.png` — SHAP summary bar plot per class

- Logs:
  - `logs/churn_library.log` — timestamped test results

---

## Notes

> Implementation notes and assumptions:

- **Logging**: configured once at import time in
  `churn_script_logging_and_tests.py` using `logging.basicConfig` with
  `filemode="w"`, an INFO level, and a timestamped format. Tests log through
  the `logging` module only; nothing is printed to stdout except the final
  summary line.
- **Plotting**: `matplotlib.use("Agg")` is set before importing `pyplot`, so
  every figure is rendered off-screen and persisted with `plt.savefig` followed
  by `plt.close`. No interactive window is ever opened, which makes the
  pipeline safe for CI and remote runs.
- **Dataset assumptions**: the CSV must contain `Attrition_Flag` and the 19
  feature columns listed in `perform_feature_engineering`; missing columns
  raise a `KeyError` rather than being silently skipped.
- **Environment**: the project targets a Windows machine with an NVIDIA GPU.
  The pipeline itself is CPU-bound (scikit-learn `RandomForestClassifier` and
  `LogisticRegression`), so no CUDA build is required; packages are installed
  without forcing CPU-only variants where a GPU-aware wheel is available.
- **Model limitations**: the dataset is imbalanced (~16% churners), so accuracy
  alone is misleading — the classification reports include per-class precision
  and recall, and the ROC curves give a threshold-independent view of ranking
  quality. Predictions describe correlations in historical data and should not
  be treated as causal explanations of churn.
- **Reproducibility**: the train/test split and the Random Forest use
  `random_state=42`; changing it will change the reported metrics.
