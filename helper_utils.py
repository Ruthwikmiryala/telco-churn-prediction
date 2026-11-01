import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score


def show_results(model, X_test, y_test):
    """Print the main classification metrics and draw the confusion matrix."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Churn"])
    plt.show()


def plot_importance(model, columns, top=10):
    """Bar chart of the most important features."""
    imp = pd.Series(model.feature_importances_, index=columns).nlargest(top)
    imp.sort_values().plot(kind="barh", title="Top features")
    plt.tight_layout()
    plt.show()
