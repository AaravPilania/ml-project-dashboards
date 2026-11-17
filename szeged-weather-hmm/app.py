import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import weather_model as w
from ui import setup, overview, cm_fig, ACCENT

setup("Szeged Weather · HMM", "🌦️")
SUN, RAIN = "#F59E0B", "#4F46E5"


@st.cache_data
def get_raw():
    return w.load_raw()


@st.cache_data
def get_daily():
    return w.make_daily(get_raw())


@st.cache_resource
def get_model():
    return w.train_all(get_daily())


raw = get_raw()
daily = get_daily()
m = get_model()
hmm = m["hmm"]

st.title("🌦️ Szeged Weather · Hidden Markov Model")
st.caption("Hourly Szeged weather 2006–2016 (Kaggle: budincsevity/szeged-weather) → daily features → "
           "Sunny/Rainy label (≥40% wet/overcast hours) → 18 discrete observation symbols "
           "(humidity × visibility × pressure change) → 2-state HMM with Laplace smoothing")

with st.sidebar:
    st.header("Time window")
    dmin, dmax = raw.timestamp.min().date(), raw.timestamp.max().date()
    rng = st.slider("Date range", min_value=dmin, max_value=dmax, value=(pd.Timestamp("2015-01-01").date(), dmax),
                    format="YYYY-MM-DD")
    agg = st.radio("Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    metric = st.selectbox("Variable", ["Temperature (C)", "Apparent Temperature (C)", "Humidity",
                                       "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)"])

t1, t2, t3, t4 = st.tabs(["Overview", "Explore", "HMM model & metrics", "Forecast"])

with t1:
    overview(raw.drop(columns=["timestamp"]), "weatherHistory.csv")
    c1, c2, c3 = st.columns(3)
    c1.metric("Days", f"{len(daily):,}")
    c2.metric("Sunny days", f"{(1 - daily.state.mean()):.1%}")
    c3.metric("Rainy days", f"{daily.state.mean():.1%}")

with t2:
    win = raw[(raw.timestamp.dt.date >= rng[0]) & (raw.timestamp.dt.date <= rng[1])].sort_values("timestamp")
    win = win.assign(**{"Pressure (millibars)": win["Pressure (millibars)"].replace(0, np.nan)})
    rule = {"Hourly": None, "Daily": "D", "Weekly": "W", "Monthly": "MS"}[agg]
    series = win.set_index("timestamp")[metric]
    if rule:
        series = series.resample(rule).agg(["mean", "min", "max"])
    else:
        series = series.to_frame("mean")
    last = series["mean"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Latest {agg.lower()} value", f"{last.iloc[-1]:.2f}" if len(last) else "–",
              f"{last.iloc[-1] - last.iloc[-2]:+.2f}" if len(last) > 1 else None)
    c2.metric("Mean in window", f"{last.mean():.2f}")
    c3.metric("Min / Max", f"{last.min():.1f} / {last.max():.1f}")
    c4.metric("Hourly records", f"{len(win):,}")
    fig = go.Figure()
    if "min" in series:
        fig.add_trace(go.Scatter(x=series.index, y=series["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=series.index, y=series["min"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(79,70,229,0.12)", name="min–max range"))
    fig.add_trace(go.Scatter(x=series.index, y=series["mean"], line=dict(color=ACCENT, width=2), name="mean"))
    fig.update_layout(title=f"{metric} · {agg.lower()} · {rng[0]} → {rng[1]}", hovermode="x unified",
                      xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    summ = win["Summary"].value_counts().head(10).reset_index()
    a.plotly_chart(px.bar(summ, x="count", y="Summary", orientation="h", title="Top weather summaries in window")
                   .update_yaxes(categoryorder="total ascending"), width="stretch")
    dwin = daily.loc[str(rng[0]):str(rng[1])]
    ms = dwin.state.resample("MS").mean().reset_index()
    b.plotly_chart(px.bar(ms, x="date" if "date" in ms else ms.columns[0], y="state",
                          title="Share of Rainy-state days per month", labels={"state": "Rainy share"}),
                   width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(win.sample(min(len(win), 4000), random_state=0), x="Temperature (C)",
                              y="Humidity", color="Precip Type", opacity=0.5,
                              title="Temperature vs humidity (hourly sample)"), width="stretch")
    cc = ["humidity", "visibility", "pressure", "pressure_change", "wet_frac", "state"]
    b.plotly_chart(px.imshow(dwin[cc].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Correlation of daily features (window)"), width="stretch")

with t3:
    res = m["results"]
    tuned = res.iloc[1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train / Val / Test days", f"{len(m['train'])} / {len(m['val'])} / {len(m['test'])}")
    c2.metric("HMM accuracy (tuned)", f"{tuned['Accuracy']:.3f}")
    c3.metric("Balanced accuracy", f"{tuned['Balanced acc']:.3f}")
    c4.metric("ROC-AUC", f"{tuned['ROC-AUC']:.3f}")
    c5.metric("Tuned threshold", f"{m['best_th']:.2f}")
    st.markdown("**Next-day prediction on the test period** (threshold tuned on validation only)")
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#E0E7FF"), width="stretch")
    st.caption(f"Majority-class accuracy: {m['majority']:.3f}")

    a, b, c = st.columns(3)
    A = pd.DataFrame(hmm.A, index=w.STATES, columns=w.STATES)
    a.plotly_chart(px.imshow(A.round(3), text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=1,
                             labels=dict(x="Tomorrow", y="Today"), title="Transition matrix A"), width="stretch")
    b.plotly_chart(cm_fig(m["cm"], w.STATES, "Confusion matrix (next day, tuned)"), width="stretch")
    pi = pd.DataFrame({"State": w.STATES, "p": hmm.pi})
    c.plotly_chart(px.bar(pi, x="State", y="p", color="State", color_discrete_sequence=[SUN, RAIN],
                          title="Start probabilities π").update_layout(showlegend=False), width="stretch")
    st.plotly_chart(px.imshow(pd.DataFrame(hmm.B, index=w.STATES).round(2), text_auto=True, aspect="auto",
                              color_continuous_scale="YlGnBu", labels=dict(x="Observation symbol", y="State"),
                              title="Emission matrix B (18 symbols)"), width="stretch")

    st.subheader("Hidden-state sequence: actual vs Viterbi vs forward filter")
    test = m["test"]
    n = st.slider("Test days shown", 30, len(test), 120, step=10)
    seqdf = pd.DataFrame({"date": test.index[:n], "Actual": test.state.values[:n],
                          "Viterbi": m["viterbi_test"][:n], "P(Rainy) filtered": m["alpha_test"][:n, 1]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seqdf.date, y=seqdf["P(Rainy) filtered"], fill="tozeroy", name="P(Rainy) (forward)",
                             line=dict(color="rgba(79,70,229,0.35)")))
    fig.add_trace(go.Scatter(x=seqdf.date, y=seqdf.Actual, line_shape="hv", name="Actual", line=dict(color="#111827", width=2)))
    fig.add_trace(go.Scatter(x=seqdf.date, y=seqdf.Viterbi, line_shape="hv", name="Viterbi",
                             line=dict(color=SUN, width=2, dash="dash")))
    fig.update_layout(yaxis=dict(tickvals=[0, 1], ticktext=w.STATES), hovermode="x unified",
                      title="0 = Sunny, 1 = Rainy")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Viterbi decoding accuracy on the full test period: "
               f"{(m['viterbi_test'] == test.state.values).mean():.3f}")

with t4:
    st.markdown("Describe **today** and the HMM updates its belief, then predicts tomorrow and the next 7 days.")
    e = m["edges"]
    a, b, c, d = st.columns(4)
    yest = a.selectbox("Yesterday was", ["Sunny", "Rainy", "Unknown"])
    hum = b.slider("Avg humidity today", 0.3, 1.0, float(round(daily.humidity.median(), 2)), 0.01)
    vis = c.slider("Avg visibility today (km)", 0.0, 16.0, float(round(daily.visibility.median(), 1)), 0.1)
    dp = d.slider("Pressure change vs yesterday (mb)", -20.0, 20.0, 0.0, 0.5)
    sym = w.symbol_for(hum, vis, dp, e)
    prior = hmm.pi if yest == "Unknown" else np.eye(2)[w.STATES.index(yest)] @ hmm.A
    belief = prior * hmm.B[:, sym]; belief /= belief.sum()
    fc = hmm.forecast(belief, 7)
    tomorrow = fc[0]
    th = m["best_th"]
    pred = "Rainy" if tomorrow[1] >= th else "Sunny"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observation symbol", sym)
    c2.metric("P(today is Rainy)", f"{belief[1]:.1%}")
    c3.metric("Tomorrow", ("🌧️ " if pred == "Rainy" else "☀️ ") + pred)
    c4.metric("P(Rainy tomorrow)", f"{tomorrow[1]:.1%}", help=f"Rainy if ≥ tuned threshold {th:.2f}")
    fdf = pd.DataFrame(fc, columns=w.STATES, index=[f"Day +{i}" for i in range(1, 8)])
    fig = go.Figure([go.Bar(x=fdf.index, y=fdf.Sunny, name="Sunny", marker_color=SUN),
                     go.Bar(x=fdf.index, y=fdf.Rainy, name="Rainy", marker_color=RAIN)])
    fig.update_layout(barmode="stack", yaxis=dict(range=[0, 1], title="Probability"),
                      title="7-day weather-state forecast")
    st.plotly_chart(fig, width="stretch")
    st.caption("The forecast drifts towards the chain's long-run balance as uncertainty grows, as in the notebook.")
