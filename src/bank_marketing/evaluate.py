from __future__ import annotations

import json
from pathlib import Path
import os

Path(".matplotlib-cache").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model, X_train, y_train, X_test, y_test) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    return {
        "train": {
            "accuracy": accuracy_score(y_train, train_pred),
            "precision": precision_score(y_train, train_pred, zero_division=0),
            "recall": recall_score(y_train, train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, train_prob),
        },
        "test": {
            "accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_prob),
            "classification_report": classification_report(
                y_test, test_pred, output_dict=True, zero_division=0
            ),
        },
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_confusion_matrix(model, X_test, y_test, path: str | Path) -> None:
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_roc_curve(model, X_test, y_test, path: str | Path) -> None:
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(model, path: str | Path, top_n: int = 20) -> None:
    if not hasattr(model, "named_steps"):
        return
    classifier = model.named_steps.get("classifier")
    columns = model.named_steps.get("columns")
    preprocessor = model.named_steps.get("preprocessor")
    if columns is None and preprocessor is not None:
        columns = preprocessor.named_steps.get("columns")
    if classifier is None or columns is None:
        return
    if not hasattr(classifier, "feature_importances_"):
        return

    names = columns.get_feature_names_out()
    importance = pd.DataFrame(
        {"feature": names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance.head(top_n), x="importance", y="feature", color="teal")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
