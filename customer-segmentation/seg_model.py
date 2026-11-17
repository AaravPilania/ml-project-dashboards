"""Customer segmentation: KMeans (k=4) on scaled + one-hot features (faithful to customer_segmentation.ipynb)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples

DATA = Path(__file__).resolve().parent / "data" / "Test.csv"
NUM = ["Age", "Work_Experience", "Family_Size"]
CAT = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
SPEND = {"Low": 0, "Average": 1, "High": 2}


def load_raw():
    return pd.read_csv(DATA)


def preprocess(df):
    data = df.drop(columns=["ID"]).copy()
    fills = {}
    for c in NUM:
        fills[c] = data[c].median(); data[c] = data[c].fillna(fills[c])
    for c in CAT:
        fills[c] = data[c].mode()[0]; data[c] = data[c].fillna(fills[c])
    data["Spending_Score"] = data["Spending_Score"].map(SPEND)
    onehot = [c for c in CAT if c != "Spending_Score"]
    X = pd.get_dummies(data, columns=onehot, drop_first=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return data, X, Xs, scaler, fills, onehot


def k_sweep(Xs, ks=range(2, 11)):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(Xs)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(Xs, lab),
                     "Davies-Bouldin": davies_bouldin_score(Xs, lab),
                     "Calinski-Harabasz": calinski_harabasz_score(Xs, lab)})
    return pd.DataFrame(rows).set_index("k")


def fit(Xs, k=4):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    lab = km.labels_
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    return dict(km=km, labels=lab, pca=pca, coords=pca.transform(Xs),
                sil=silhouette_score(Xs, lab), db=davies_bouldin_score(Xs, lab),
                ch=calinski_harabasz_score(Xs, lab), sil_samples=silhouette_samples(Xs, lab))


def profile(df, labels):
    d = df.assign(Cluster=labels)
    g = d.groupby("Cluster")
    p = g.agg(Size=("ID", "count"), Avg_Age=("Age", "mean"),
              Avg_Work_Exp=("Work_Experience", "mean"), Avg_Family=("Family_Size", "mean")).round(1)
    p["Top_Profession"] = g["Profession"].agg(lambda s: s.mode()[0])
    p["Top_Spending"] = g["Spending_Score"].agg(lambda s: s.mode()[0])
    p["Married_%"] = g["Ever_Married"].apply(lambda s: (s == "Yes").mean() * 100).round(0)
    p["Graduated_%"] = g["Graduated"].apply(lambda s: (s == "Yes").mean() * 100).round(0)
    age = lambda a: "Young" if a < 35 else ("Middle-aged" if a < 55 else "Senior")
    p["Persona"] = [f"{age(r.Avg_Age)} {r.Top_Spending}-spenders" for r in p.itertuples()]
    return p


def encode_one(row: dict, X_columns, onehot):
    d = pd.DataFrame([row])
    d["Spending_Score"] = d["Spending_Score"].map(SPEND)
    d = pd.get_dummies(d, columns=onehot)
    return d.reindex(columns=X_columns, fill_value=0).astype(float)
