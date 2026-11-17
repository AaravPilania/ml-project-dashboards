"""Breast cancer: Decision tree vs Bagging(100 trees) (faithful to bagging_decision_tree_breast_cancer.ipynb)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

DATA = Path(__file__).resolve().parent / "data" / "data.csv"
DROP_CORR = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se',
             'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
             'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']


def load():
    data = pd.read_csv(DATA)
    y = data.diagnosis
    x = data.drop([c for c in ['Unnamed: 32', 'id', 'diagnosis'] if c in data.columns], axis=1)
    return data, x, y


def bag(n=100, oob=False, n_jobs=-1):
    return BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=n,
                             bootstrap=True, oob_score=oob, n_jobs=n_jobs, random_state=42)


def train(x, y, n_estimators=100):
    y_num = y.map({'M': 1, 'B': 0})
    xtr, xte, ytr, yte = train_test_split(x, y_num, test_size=0.3, stratify=y_num, random_state=42)
    dt = DecisionTreeClassifier(random_state=42).fit(xtr, ytr)
    bg = bag(n_estimators, oob=True).fit(xtr, ytr)
    out = {"xtr": xtr, "xte": xte, "ytr": ytr, "yte": yte, "dt": dt, "bag": bg, "y_num": y_num}
    for name, m in [("dt", dt), ("bag", bg)]:
        pred = m.predict(xte); proba = m.predict_proba(xte)[:, 1]
        out[name + "_m"] = {
            "train_acc": accuracy_score(ytr, m.predict(xtr)), "test_acc": accuracy_score(yte, pred),
            "auc": roc_auc_score(yte, proba), "cm": confusion_matrix(yte, pred),
            "report": pd.DataFrame(classification_report(yte, pred, target_names=["Benign", "Malignant"],
                                                         output_dict=True)).T,
            "roc": roc_curve(yte, proba)}
    out["bag_m"]["oob"] = bg.oob_score_
    out["importance"] = pd.Series(np.mean([t.feature_importances_ for t in bg.estimators_], axis=0),
                                  index=x.columns).sort_values(ascending=False)
    # correlation-reduced variant (notebook section 11)
    x1 = x.drop(DROP_CORR, axis=1)
    a, b, _, _ = train_test_split(x1, y_num, test_size=0.3, stratify=y_num, random_state=42)
    out["x1_acc"] = {"Decision Tree": accuracy_score(yte, DecisionTreeClassifier(random_state=42).fit(a, ytr).predict(b)),
                     "Bagging": accuracy_score(yte, bag(100).fit(a, ytr).predict(b))}
    return out


def cv_compare(x, y):
    y_num = y.map({'M': 1, 'B': 0})
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    dt_cv = cross_val_score(DecisionTreeClassifier(random_state=42), x, y_num, cv=cv, scoring="accuracy", n_jobs=-1)
    bag_cv = cross_val_score(bag(100, n_jobs=None), x, y_num, cv=cv, scoring="accuracy", n_jobs=-1)
    return dt_cv, bag_cv


def n_trees_curve(x, y, ns=(1, 5, 10, 25, 50, 100, 200)):
    y_num = y.map({'M': 1, 'B': 0}); rows = []
    for n in ns:
        s = cross_val_score(bag(n), x, y_num, cv=5, scoring="accuracy")
        rows.append({"n_estimators": n, "mean": s.mean(), "std": s.std()})
    return pd.DataFrame(rows)
