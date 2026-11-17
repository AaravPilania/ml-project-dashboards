import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import cancer_model as bc
from ui import setup, overview, cm_fig

setup("Breast Cancer · Bagging", "🩺")
B_COL, M_COL = "#10B981", "#EF4444"
CMAP = {"B": B_COL, "M": M_COL}


@st.cache_data
def get_data():
    return bc.load()


@st.cache_resource
def get_model(n):
    data, x, y = get_data()
    return bc.train(x, y, n)


@st.cache_data
def get_cv():
    _, x, y = get_data()
    return bc.cv_compare(x, y)


@st.cache_data
def get_curve():
    _, x, y = get_data()
    return bc.n_trees_curve(x, y)


data, x, y = get_data()

st.title("🩺 Breast Cancer Wisconsin · Bagged Decision Trees")
st.caption("WDBC, 569 tumours × 30 features (Kaggle: uciml/breast-cancer-wisconsin-data) · stratified 70/30 split "
           "(random_state 42) · single Decision Tree vs Bagging of 100 trees with OOB score")

with st.sidebar:
    st.header("Filters (EDA)")
    diag = st.multiselect("Diagnosis", ["B", "M"], default=["B", "M"],
                          format_func=lambda d: "Benign (B)" if d == "B" else "Malignant (M)")
    group = st.radio("Feature group", ["mean", "se", "worst"], horizontal=True)
    st.divider()
    st.header("Model")
    n_est = st.select_slider("Bagging: number of trees", [1, 5, 10, 25, 50, 100, 200], value=100,
                             help="The notebook uses 100")

m = get_model(n_est)
dm, bm = m["dt_m"], m["bag_m"]
t1, t2, t3, t4 = st.tabs(["Overview", "Explore", "Model & metrics", "Predict a tumour"])

with t1:
    overview(data, "data.csv")
    c1, c2 = st.columns(2)
    c1.metric("Benign", int((y == "B").sum()))
    c2.metric("Malignant", int((y == "M").sum()))

with t2:
    sel = data[data.diagnosis.isin(diag)]
    cols = [c for c in x.columns if c.endswith("_" + group)]
    a, b = st.columns([1, 2])
    a.plotly_chart(px.pie(sel, names="diagnosis", hole=0.55, color="diagnosis", color_discrete_map=CMAP,
                          title="Diagnosis split"), width="stretch")
    z = (x[cols] - x[cols].mean()) / x[cols].std()
    melt = pd.concat([y, z], axis=1).loc[sel.index].melt(id_vars="diagnosis", var_name="feature", value_name="value")
    b.plotly_chart(px.violin(melt, x="feature", y="value", color="diagnosis", color_discrete_map=CMAP, box=True,
                             title=f"Standardised '{group}' features by diagnosis").update_xaxes(tickangle=35),
                   width="stretch")
    a, b, c = st.columns([1, 1, 2])
    fx = a.selectbox("X feature", x.columns, index=list(x.columns).index("radius_mean"))
    fy = b.selectbox("Y feature", x.columns, index=list(x.columns).index("concave points_mean"))
    st.plotly_chart(px.scatter(sel, x=fx, y=fy, color="diagnosis", color_discrete_map=CMAP, opacity=0.7,
                               marginal_x="histogram", marginal_y="histogram", title=f"{fx} vs {fy}"),
                    width="stretch")
    st.plotly_chart(px.imshow(x[cols].corr().round(1), text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title=f"Correlation of '{group}' features"), width="stretch")

with t3:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tree test accuracy", f"{dm['test_acc']:.3f}", f"train {dm['train_acc']:.3f}", delta_color="off")
    c2.metric(f"Bagging test accuracy ({n_est})", f"{bm['test_acc']:.3f}", f"{bm['test_acc'] - dm['test_acc']:+.3f} vs tree")
    c3.metric("Bagging OOB score", f"{bm['oob']:.3f}" if n_est >= 10 else "n/a")
    c4.metric("Tree ROC-AUC", f"{dm['auc']:.3f}")
    c5.metric("Bagging ROC-AUC", f"{bm['auc']:.3f}")

    a, b, c = st.columns(3)
    a.plotly_chart(cm_fig(dm["cm"], ["Benign", "Malignant"], "Decision Tree"), width="stretch")
    b.plotly_chart(cm_fig(bm["cm"], ["Benign", "Malignant"], "Bagging"), width="stretch")
    fig = go.Figure()
    for name, mm, col in [("Decision Tree", dm, "#94A3B8"), ("Bagging", bm, "#4F46E5")]:
        fpr, tpr, _ = mm["roc"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC {mm['auc']:.3f})", line=dict(color=col, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#CBD5E1"), showlegend=False))
    fig.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate",
                      legend=dict(x=0.35, y=0.05))
    c.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Classification report · Decision Tree**")
        st.dataframe(dm["report"].style.format("{:.3f}"), width="stretch")
    with b:
        st.markdown("**Classification report · Bagging**")
        st.dataframe(bm["report"].style.format("{:.3f}"), width="stretch")

    imp = m["importance"].head(10).sort_values()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": ""},
                          title="Top 10 features (averaged over bagged trees)"), width="stretch")
    x1 = pd.DataFrame({"Model": list(m["x1_acc"]), "Accuracy": list(m["x1_acc"].values())})
    b.plotly_chart(px.bar(x1, x="Model", y="Accuracy", text_auto=".3f", range_y=[0.8, 1],
                          title="Same models on correlation-reduced features (16 cols)"), width="stretch")

    st.subheader("Variance reduction")
    if st.toggle("Run 25-fold cross-validation and the n_estimators curve (~10 s)"):
        dt_cv, bag_cv = get_cv()
        cvdf = pd.DataFrame({"Decision Tree": dt_cv, "Bagging (100 trees)": bag_cv}).melt(var_name="Model", value_name="Accuracy")
        a, b = st.columns(2)
        a.plotly_chart(px.box(cvdf, x="Model", y="Accuracy", color="Model", points="all",
                              title="Accuracy spread over 25 folds (smaller box = lower variance)")
                       .update_layout(showlegend=False), width="stretch")
        cur = get_curve()
        fig = go.Figure([
            go.Scatter(x=cur.n_estimators, y=cur["mean"] + cur["std"], line=dict(width=0), showlegend=False),
            go.Scatter(x=cur.n_estimators, y=cur["mean"] - cur["std"], fill="tonexty", line=dict(width=0),
                       fillcolor="rgba(79,70,229,0.15)", name="±1 std"),
            go.Scatter(x=cur.n_estimators, y=cur["mean"], mode="lines+markers", name="Mean CV accuracy",
                       line=dict(color="#4F46E5"))])
        fig.update_layout(title="Bagging: accuracy vs number of trees", xaxis_type="log", xaxis_title="n_estimators")
        b.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame({"Model": ["Decision Tree", "Bagging (100 trees)"],
                                   "Mean acc": [dt_cv.mean(), bag_cv.mean()],
                                   "Std": [dt_cv.std(), bag_cv.std()],
                                   "Variance": [dt_cv.var(), bag_cv.var()]}).style.format(precision=4),
                     hide_index=True, width="stretch")

with t4:
    st.markdown("Start from a real test-set tumour or the dataset median, then adjust the most important features.")
    a, b = st.columns([1, 2])
    src = a.radio("Start from", ["Dataset median", "A test-set tumour"])
    base = x.median()
    if src == "A test-set tumour":
        idx = b.selectbox("Tumour (row id · true label)", m["xte"].index,
                          format_func=lambda i: f"#{i} · {'Malignant' if y[i] == 'M' else 'Benign'}")
        base = x.loc[idx]
    top = list(m["importance"].head(8).index)
    vals = base.copy()
    cols = st.columns(4)
    for i, f in enumerate(top):
        lo, hi = float(x[f].min()), float(x[f].max())
        vals[f] = cols[i % 4].slider(f, lo, hi, float(base[f]), (hi - lo) / 200, key=f"s_{f}_{src}_{base.name}")
    xi = vals.to_frame().T[x.columns]
    p_bag = m["bag"].predict_proba(xi)[0, 1]
    p_dt = m["dt"].predict_proba(xi)[0, 1]
    votes = np.mean([t.predict(xi.values)[0] for t in m["bag"].estimators_])
    lab = "Malignant" if p_bag >= 0.5 else "Benign"
    c1, c2, c3 = st.columns(3)
    c1.metric("Bagging prediction", ("🔴 " if lab == "Malignant" else "🟢 ") + lab)
    c2.metric("P(Malignant) · bagging", f"{p_bag:.1%}")
    c3.metric("Single tree says", "Malignant" if p_dt >= 0.5 else "Benign")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p_bag * 100, number={"suffix": "%"},
                                 title={"text": f"Malignant probability ({len(m['bag'].estimators_)} trees, {votes:.0%} vote malignant)"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": M_COL if lab == "Malignant" else B_COL},
                                        "steps": [{"range": [0, 50], "color": "#ECFDF5"}, {"range": [50, 100], "color": "#FEF2F2"}],
                                        "threshold": {"line": {"color": "#111827", "width": 3}, "value": 50}}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")
    st.caption("Educational demo only, not a medical tool.")
