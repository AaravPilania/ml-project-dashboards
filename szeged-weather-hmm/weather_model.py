"""Szeged weather: daily Sunny/Rainy labels + discrete HMM (faithful to weather_prediction.ipynb)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)

DATA = Path(__file__).resolve().parent / "data" / "weatherHistory.csv"
STATES = ["Sunny", "Rainy"]
WET_THRESHOLD = 0.4
N_OBS = 18
FEAT = ["humidity", "visibility", "pressure", "pressure_change", "temperature", "wind", "wet_frac"]


def load_raw():
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["Formatted Date"].str[:19])
    return df


def make_daily(df, wet_threshold=WET_THRESHOLD):
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["Pressure (millibars)"] = df["Pressure (millibars)"].replace(0, np.nan)
    df["wet_hour"] = df["Summary"].str.contains("Rain|Drizzle|Overcast|Foggy|Humid", case=False, na=False)
    g = df.groupby("date")
    daily = pd.DataFrame({
        "humidity": g["Humidity"].mean(),
        "visibility": g["Visibility (km)"].mean(),
        "pressure": g["Pressure (millibars)"].mean(),
        "temperature": g["Temperature (C)"].mean(),
        "wind": g["Wind Speed (km/h)"].mean(),
        "wet_frac": g["wet_hour"].mean(),
    }).sort_index()
    for c in ["pressure", "temperature"]:
        daily[c] = daily[c].interpolate().bfill().ffill()
    daily["pressure_change"] = daily["pressure"].diff().fillna(0)
    daily["state"] = (daily["wet_frac"] >= wet_threshold).astype(int)
    daily.index = pd.to_datetime(daily.index)
    return daily


class HMM:
    def fit(self, states, obs, n_obs, alpha=1.0):
        K = len(STATES); self.K = K
        self.pi = np.full(K, alpha); self.pi[states[0]] += 1; self.pi /= self.pi.sum()
        A = np.full((K, K), alpha)
        for a, b in zip(states[:-1], states[1:]):
            A[a, b] += 1
        self.A = A / A.sum(1, keepdims=True)
        B = np.full((K, n_obs), alpha)
        for s, o in zip(states, obs):
            B[s, o] += 1
        self.B = B / B.sum(1, keepdims=True)
        return self

    def forward(self, obs):
        T = len(obs); alpha = np.zeros((T, self.K))
        a = self.pi * self.B[:, obs[0]]; alpha[0] = a / a.sum()
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]; alpha[t] = a / a.sum()
        return alpha

    def viterbi(self, obs):
        T = len(obs)
        lA, lB, lpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = np.zeros((T, self.K)); psi = np.zeros((T, self.K), dtype=int)
        delta[0] = lpi + lB[:, obs[0]]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + lA
            psi[t] = cand.argmax(0); delta[t] = cand.max(0) + lB[:, obs[t]]
        path = np.zeros(T, dtype=int); path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def forecast(self, belief, days):
        out, b = [], belief
        for _ in range(days):
            b = b @ self.A; out.append(b)
        return np.array(out)


def train_all(daily):
    n = len(daily); i1, i2 = int(n * 0.70), int(n * 0.85)
    train, val, test = daily.iloc[:i1].copy(), daily.iloc[i1:i2].copy(), daily.iloc[i2:].copy()
    edges = {
        "h": np.unique(train["humidity"].quantile([1 / 3, 2 / 3]).values),
        "v": np.unique(train["visibility"].quantile([1 / 3, 2 / 3]).values),
        "p": np.unique(train["pressure_change"].quantile([0.5]).values),
    }

    def encode(d):
        h = np.digitize(d["humidity"], edges["h"]); v = np.digitize(d["visibility"], edges["v"])
        p = np.digitize(d["pressure_change"], edges["p"])
        return ((h * 3 + v) * 2 + p).astype(int)

    obs = {k: encode(d) for k, d in [("train", train), ("val", val), ("test", test)]}
    st = {k: d["state"].values for k, d in [("train", train), ("val", val), ("test", test)]}
    hmm = HMM().fit(st["train"], obs["train"], N_OBS)

    def next_day(o, s):
        a = hmm.forward(o)
        return (a[:-1] @ hmm.A)[:, 1], s[1:]

    p_val, y_val = next_day(obs["val"], st["val"])
    p_test, y_test = next_day(obs["test"], st["test"])
    ths = np.linspace(0.05, 0.95, 91)
    best_th = float(ths[np.argmax([balanced_accuracy_score(y_val, p_val >= t) for t in ths])])

    def xy(d):
        X = d[FEAT].copy(); X["state_today"] = d["state"]
        return X.iloc[:-1].values, d["state"].values[1:]
    Xtr, ytr = xy(train); Xte, yte = xy(test)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    p_lr = lr.predict_proba((Xte - mu) / sd)[:, 1]

    def report(name, y, p, th=0.5):
        pred = (p >= th).astype(int)
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
        return {"Model": name, "Accuracy": accuracy_score(y, pred),
                "Balanced acc": balanced_accuracy_score(y, pred),
                "Rainy precision": precision_score(y, pred, zero_division=0),
                "Rainy recall": recall_score(y, pred, zero_division=0),
                "Rainy F1": f1_score(y, pred, zero_division=0), "ROC-AUC": auc}

    persist = st["test"][:-1]
    results = pd.DataFrame([
        report("HMM (th=0.5)", y_test, p_test, 0.5),
        report(f"HMM (tuned th={best_th:.2f})", y_test, p_test, best_th),
        report("Persistence", y_test, persist.astype(float), 0.5),
        report("Logistic regression", yte, p_lr, 0.5),
    ]).set_index("Model")
    alpha = hmm.forward(obs["test"])
    return dict(hmm=hmm, edges=edges, encode=encode, train=train, val=val, test=test, obs=obs, st=st,
                p_test=p_test, y_test=y_test, best_th=best_th, results=results,
                majority=float(max(y_test.mean(), 1 - y_test.mean())),
                alpha_test=alpha, viterbi_test=hmm.viterbi(obs["test"]),
                cm=confusion_matrix(y_test, (p_test >= best_th).astype(int), labels=[0, 1]))


def symbol_for(humidity, visibility, pressure_change, edges):
    h = int(np.digitize([humidity], edges["h"])[0]); v = int(np.digitize([visibility], edges["v"])[0])
    p = int(np.digitize([pressure_change], edges["p"])[0])
    return (h * 3 + v) * 2 + p
