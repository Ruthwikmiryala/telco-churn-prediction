# Telco Customer Churn (Random Forest)

Predict whether a telecom customer will leave.

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this folder
- Model: Random Forest, compared with a Logistic Regression baseline

## Files

| File | What it does |
|---|---|
| `TelcoFeatures.py` | `clean_data`, `add_features`, `encode` |
| `helper_utils.py` | `show_results`, `plot_importance` |
| `Telco_churn_analysis.ipynb` | Data exploration |
| `Telco_Churn_Final.ipynb` | Model training and evaluation |
| `tests/test_telco_features.py` | Unit tests |

## Run

```bash
pip install -r requirements.txt
pytest
jupyter lab        # open the notebooks from this folder
```

## Results

Fill in from your run of `Telco_Churn_Final.ipynb`:

| Model | ROC-AUC | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |
