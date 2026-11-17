import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seg_model as seg
from ui import setup, overview, PALETTE

setup("Customer Segmentation", "🧩")


@st.cache_data
def get_data():
    df = seg.load_raw()
    data, X, Xs, scaler, fills, onehot = seg.preprocess(df)
    return df, data, X, Xs, scaler, onehot


@st.cache_data
def get_sweep(Xs):
    return seg.k_sweep(Xs)


@st.cache_resource
def get_model(_Xs, k):
    return seg.fit(_Xs, k)


df, data, X, Xs, scaler, onehot = get_data()

st.title("🧩 Customer Segmentation")
st.caption("K-Means clustering of customers (Kaggle: abisheksudarshan/customer-segmentation, Test.csv) · "
           "median/mode imputation → ordinal Spending_Score → one-hot → StandardScaler → K-Means")

with st.sidebar:
    st.header("Filters (EDA)")
    genders = st.multiselect("Gender", sorted(df.Gender.dropna().unique()), default=sorted(df.Gender.dropna().unique()))
    spend = st.multiselect("Spending score", ["Low", "Average", "High"], default=["Low", "Average", "High"])
    profs = sorted(df.Profession.dropna().unique())
    prof = st.multiselect("Profession", profs, default=profs)
    age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))
    st.divider()
    st.header("Model")
    k = st.slider("Number of clusters (k)", 2, 10, 4, help="The notebook uses k = 4")

mask = (df.Gender.isin(genders) & df.Spending_Score.isin(spend) & (df.Profession.isin(prof) | df.Profession.isna())
        & df.Age.between(*age))
f = df[mask]

m = get_model(Xs, k)
labels = m["labels"]

t1, t2, t3, t4 = st.tabs(["Overview", "Explore", "Clusters & metrics", "Predict a customer"])

with t1:
    overview(df, "Test.csv")

with t2:
    st.markdown(f"**{len(f):,}** of {len(df):,} customers match the filters.")
    c1, c2, c3 = st.columns(3)
    for col, c in zip(seg.NUM, [c1, c2, c3]):
        c.plotly_chart(px.histogram(f, x=col, nbins=30, title=col), width="stretch")
    cat = st.selectbox("Categorical breakdown", seg.CAT, index=3)
    a, b = st.columns(2)
    vc = f[cat].value_counts().reset_index()
    a.plotly_chart(px.bar(vc, x=cat, y="count", title=f"{cat} counts"), width="stretch")
    b.plotly_chart(px.box(f, x=cat, y="Age", title=f"Age by {cat}"), width="stretch")
    fd = data.loc[f.index, ["Age", "Work_Experience", "Family_Size", "Spending_Score"]]
    st.plotly_chart(px.imshow(fd.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation (numeric + ordinal Spending_Score)"),
                    width="stretch")

with t3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters (k)", k)
    c2.metric("Silhouette ↑", f"{m['sil']:.3f}")
    c3.metric("Davies-Bouldin ↓", f"{m['db']:.3f}")
    c4.metric("Calinski-Harabasz ↑", f"{m['ch']:.1f}")

    sweep = get_sweep(Xs)
    a, b = st.columns(2)
    fig = px.line(sweep.reset_index(), x="k", y="Inertia", markers=True, title="Elbow method (inertia)")
    fig.add_vline(x=k, line_dash="dash", line_color="#94A3B8")
    a.plotly_chart(fig, width="stretch")
    fig = px.line(sweep.reset_index(), x="k", y="Silhouette", markers=True, title="Silhouette score by k")
    fig.add_vline(x=k, line_dash="dash", line_color="#94A3B8")
    b.plotly_chart(fig, width="stretch")

    a, b = st.columns([3, 2])
    pc = pd.DataFrame(m["coords"], columns=["PC1", "PC2"]).assign(Cluster=labels.astype(str))
    a.plotly_chart(px.scatter(pc, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                              category_orders={"Cluster": [str(i) for i in range(k)]},
                              title=f"Clusters in PCA space (k={k})"), width="stretch")
    sizes = pd.Series(labels).value_counts().sort_index()
    b.plotly_chart(px.pie(values=sizes.values, names=[f"Cluster {i}" for i in sizes.index], hole=0.55,
                          title="Cluster sizes"), width="stretch")

    st.subheader("Cluster profiles")
    prof_tbl = seg.profile(df, labels)
    st.dataframe(prof_tbl, width="stretch")

    radar_cols = [c for c in ["Age", "Work_Experience", "Family_Size", "Spending_Score",
                              "Ever_Married_Yes", "Graduated_Yes", "Gender_Male"] if c in X.columns]
    means = X.groupby(labels)[radar_cols].mean()
    means = (means - means.min()) / (means.max() - means.min() + 1e-9)
    fig = go.Figure()
    for i, (cl, row) in enumerate(means.iterrows()):
        fig.add_trace(go.Scatterpolar(r=list(row) + [row.iloc[0]], theta=radar_cols + [radar_cols[0]],
                                      fill="toself", opacity=0.55, name=f"Cluster {cl}",
                                      line_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(title="Radar chart of cluster profiles (min-max scaled)", polar=dict(radialaxis=dict(visible=False)))
    a, b = st.columns(2)
    a.plotly_chart(fig, width="stretch")
    sil = pd.DataFrame({"Cluster": labels.astype(str), "s": m["sil_samples"]})
    neg = sil.groupby("Cluster")["s"].apply(lambda s: (s < 0).mean() * 100).round(1)
    b.plotly_chart(px.box(sil, x="Cluster", y="s", title="Silhouette per point, by cluster",
                          labels={"s": "Silhouette coefficient"}), width="stretch")
    b.caption("Share of points with negative silhouette: " + ", ".join(f"C{c}: {v}%" for c, v in neg.items()))

with t4:
    st.markdown("Enter a customer's details to see which segment K-Means assigns them to.")
    with st.form("predict"):
        a, b, c = st.columns(3)
        g = a.selectbox("Gender", ["Male", "Female"])
        mar = b.selectbox("Ever married", ["Yes", "No"])
        grad = c.selectbox("Graduated", ["Yes", "No"])
        a, b, c = st.columns(3)
        ag = a.slider("Age", 18, 90, 35)
        we = b.slider("Work experience (years)", 0, 14, 1)
        fs = c.slider("Family size", 1, 9, 3)
        a, b, c = st.columns(3)
        pr = a.selectbox("Profession", profs)
        sp = b.selectbox("Spending score", ["Low", "Average", "High"])
        v1 = c.selectbox("Var_1", sorted(df.Var_1.dropna().unique()), index=5)
        go_ = st.form_submit_button("Assign segment", type="primary")
    row = {"Gender": g, "Ever_Married": mar, "Age": ag, "Graduated": grad, "Profession": pr,
           "Work_Experience": we, "Spending_Score": sp, "Family_Size": fs, "Var_1": v1}
    x1 = seg.encode_one(row, X.columns, onehot)
    xs = scaler.transform(x1)
    cl = int(m["km"].predict(xs)[0])
    d = m["km"].transform(xs)[0]
    prof_tbl = seg.profile(df, labels)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assigned cluster", f"Cluster {cl}")
    c2.metric("Persona", prof_tbl.loc[cl, "Persona"])
    c3.metric("Cluster size", f"{int(prof_tbl.loc[cl, 'Size']):,} customers")
    dist = pd.DataFrame({"Cluster": [f"Cluster {i}" for i in range(k)], "Distance": d})
    a, b = st.columns(2)
    a.plotly_chart(px.bar(dist, x="Cluster", y="Distance", title="Distance to each cluster centre (lower = closer)"),
                   width="stretch")
    p2 = m["pca"].transform(xs)[0]
    fig = px.scatter(pd.DataFrame(m["coords"], columns=["PC1", "PC2"]).assign(Cluster=labels.astype(str)),
                     x="PC1", y="PC2", color="Cluster", opacity=0.25, title="Where this customer sits (PCA)",
                     category_orders={"Cluster": [str(i) for i in range(k)]})
    fig.add_trace(go.Scatter(x=[p2[0]], y=[p2[1]], mode="markers", name="This customer",
                             marker=dict(size=18, symbol="star", color="#111827", line=dict(width=2, color="white"))))
    b.plotly_chart(fig, width="stretch")
