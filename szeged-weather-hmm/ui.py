import plotly.express as px
import plotly.io as pio
import streamlit as st

ACCENT = "#4F46E5"
PALETTE = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#06B6D4", "#8B5CF6", "#EC4899", "#84CC16", "#64748B", "#F97316"]

pio.templates["clean"] = pio.templates["plotly_white"]
pio.templates["clean"].layout.update(colorway=PALETTE, font=dict(family="Inter, sans-serif", size=13),
                                     margin=dict(l=10, r=10, t=50, b=10), title=dict(x=0.0, font=dict(size=15)))
pio.templates.default = "clean"
px.defaults.template = "clean"
px.defaults.color_discrete_sequence = PALETTE


def setup(title, icon):
    try:
        st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    except Exception:
        pass
    st.markdown("""
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;}
      [data-testid="stMetric"] {background: #F5F6FA; border: 1px solid #E6E8F0; border-radius: 12px; padding: 14px 16px;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #5B6175;}
      h1 {font-weight: 700; letter-spacing: -0.02em;}
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .stTabs [data-baseweb="tab"] {padding: 6px 14px; border-radius: 8px 8px 0 0;}
      #MainMenu, footer {visibility: hidden;}
    </style>""", unsafe_allow_html=True)


def overview(df, name):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.caption(f"Source file: `data/{name}`")
    st.dataframe(df.head(10), width="stretch", hide_index=True)
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values()
    a, b = st.columns([1, 1])
    with a:
        if len(miss):
            fig = px.bar(x=miss.values, y=miss.index, orientation="h", title="Missing values per column",
                         labels={"x": "Missing", "y": ""})
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("No missing values.")
    with b:
        st.markdown("**Column types**")
        st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), width="stretch", height=300)


def cm_fig(cm, labels, title="Confusion matrix"):
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"), title=title)
    fig.update_coloraxes(showscale=False)
    return fig
