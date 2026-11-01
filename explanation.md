# Explanation

**Problem.** Predict `Churn` (Yes/No) for each customer. About one in four customers churns, so accuracy alone is misleading; we also look at ROC-AUC, precision and recall for the churn class.

**Cleaning (`clean_data`).** Drop `customerID`; `TotalCharges` is text with a few blanks (customers with no bill yet), so convert to a number and use 0; turn `Churn` into 1/0.

**Features (`add_features`).** `n_addons` (how many extras the customer has), `avg_monthly_spend` (total paid / months), `is_month_to_month` (no long contract).

**Encoding (`encode`).** `pd.get_dummies` turns text columns into 0/1 columns.

**Split.** 80% train / 20% test, `stratify=y` so both keep the same churn rate. The test set is used once, at the end.

**Models.** Logistic Regression is the baseline. The Random Forest is a crowd of decision trees that vote; `class_weight='balanced'` gives churners more weight. `GridSearchCV` tries a few settings with 5-fold cross-validation and picks the best by ROC-AUC.

**Reading the results.** Higher recall means more churners caught; higher precision means fewer loyal customers wrongly flagged. The feature-importance chart shows which columns the forest relies on (what it uses, not what causes churn).
