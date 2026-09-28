"""
AutoWorth · Used Car Price Estimator
Streamlit frontend for the LightGBM model trained in the notebook.
Run:  streamlit run app.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(page_title="AutoWorth · Car Price Estimator", page_icon="🚘",
                   layout="wide", initial_sidebar_state="expanded")

MODEL_PATH = Path(__file__).parent / "lightgbm_car_price_model.pkl"
REF_YEAR = 2026  # same reference year used when training (car_age = 2026 - model_year)
CAT_COLS = ["brand", "fuel_type", "transmission", "ext_col", "int_col", "accident", "clean_title"]

# Approximate share of listings for a model name (the model was trained on `model_freq`)
POPULARITY = {
    "Rare (1 listing)": 0.00025,
    "Uncommon (2–5 listings)": 0.0009,
    "Common (6–20 listings)": 0.0035,
    "Very common (20+ listings)": 0.008,
}

FRIENDLY = {
    "brand": "Brand", "fuel_type": "Fuel type", "transmission": "Transmission",
    "ext_col": "Exterior colour", "int_col": "Interior colour", "accident": "Accident history",
    "clean_title": "Clean title", "milage": "Mileage", "car_age": "Car age",
    "horsepower": "Horsepower", "displacement_L": "Engine size", "model_freq": "Model popularity",
}

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1.5rem; max-width: 1250px; }

.hero {
  background: linear-gradient(120deg, #4f2bff 0%, #7c5cff 45%, #00d4ff 100%);
  border-radius: 24px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.6rem;
  box-shadow: 0 20px 50px rgba(124,92,255,.35); position: relative; overflow: hidden;
}
.hero::after { content:"🚘"; position:absolute; right:2.5rem; top:50%; transform:translateY(-50%);
  font-size:7rem; opacity:.22; }
.hero h1 { margin:0; font-size:2.5rem; font-weight:800; letter-spacing:-1px; }
.hero p  { margin:.5rem 0 0; font-size:1.05rem; opacity:.92; max-width:640px; }
.badge { display:inline-block; background:rgba(255,255,255,.18); padding:.25rem .8rem;
  border-radius:999px; font-size:.78rem; font-weight:600; margin-bottom:.8rem; backdrop-filter:blur(6px);}

.card { background:#131A2B; border:1px solid rgba(255,255,255,.07); border-radius:20px;
  padding:1.4rem 1.6rem; box-shadow:0 10px 30px rgba(0,0,0,.25); }
.sec-title { font-weight:700; font-size:1.05rem; margin:.2rem 0 .8rem; color:#c9d1ff; }

.price-card { background:linear-gradient(145deg,#161f38,#0f1526); border:1px solid rgba(124,92,255,.35);
  border-radius:22px; padding:1.6rem; text-align:center; box-shadow:0 12px 40px rgba(124,92,255,.2);}
.price-label { text-transform:uppercase; letter-spacing:.14em; font-size:.75rem; color:#9aa6d6; font-weight:600;}
.price-value { font-size:3.4rem; font-weight:800; margin:.2rem 0;
  background:linear-gradient(90deg,#7c5cff,#00d4ff); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; }
.price-range { color:#aab4de; font-size:.95rem; }

.kpi { background:#131A2B; border:1px solid rgba(255,255,255,.07); border-radius:16px;
  padding:1rem 1.1rem; text-align:center; }
.kpi .v { font-size:1.45rem; font-weight:700; color:#fff; }
.kpi .l { font-size:.75rem; color:#8f9bc9; text-transform:uppercase; letter-spacing:.08em; }

.placeholder { text-align:center; padding:3.2rem 1rem; color:#8f9bc9; }
.placeholder .big { font-size:3.5rem; }

.stFormSubmitButton > button {
  width:100%; border:0; border-radius:14px; padding:.85rem 1rem; font-weight:700; font-size:1.05rem;
  color:#fff; background:linear-gradient(90deg,#7c5cff,#00b4ff); transition:.2s;
  box-shadow:0 8px 24px rgba(124,92,255,.4);
}
.stFormSubmitButton > button:hover { transform:translateY(-2px); box-shadow:0 12px 30px rgba(0,180,255,.45); color:#fff;}
[data-testid="stForm"] { border:0; padding:0; }
[data-testid="stSidebar"] { border-right:1px solid rgba(255,255,255,.06); }
.foot { text-align:center; color:#6f7aa6; font-size:.8rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────── Model loading ─────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        booster = pickle.load(f)
    # The category lists used at training time are stored inside the LightGBM booster.
    cats = dict(zip(CAT_COLS, booster.pandas_categorical))
    return booster, cats


try:
    model, CATS = load_model()
except Exception as e:  # pragma: no cover
    st.error(f"Could not load the model file: {e}")
    st.stop()


@st.cache_data(show_spinner=False)
def read_reference_csv(file_bytes: bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    freq = df["model"].value_counts(normalize=True)
    by_brand = df.groupby("brand")["model"].agg(lambda s: sorted(s.dropna().unique())).to_dict()
    return freq.to_dict(), by_brand


def predict_price(row: dict) -> float:
    df = pd.DataFrame([row])
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c], categories=CATS[c])
    df = df[model.feature_name()]
    return float(np.expm1(model.predict(df)[0]))


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("Optional: upload the original `used_cars.csv` to pick exact models and get the most "
               "accurate *model popularity* value.")
    ref_file = st.file_uploader("Reference dataset (CSV)", type="csv")
    freq_map, models_by_brand = (None, None)
    if ref_file is not None:
        try:
            freq_map, models_by_brand = read_reference_csv(ref_file.getvalue())
            st.success(f"Loaded {len(freq_map):,} unique models")
        except Exception as e:
            st.warning(f"Couldn't read CSV: {e}")

    st.markdown("---")
    st.markdown("### 🧠 About the model")
    st.markdown(
        "- **Algorithm:** LightGBM (gradient boosting)\n"
        "- **Target:** log(price), converted back to USD\n"
        f"- **Trees:** {model.num_trees()}\n"
        f"- **Features:** {model.num_feature()}"
    )
    st.caption("Estimates are indicative and not a formal valuation.")

# ───────────────────────── Hero ─────────────────────────
st.markdown("""
<div class="hero">
  <span class="badge">⚡ AI-POWERED VALUATION</span>
  <h1>AutoWorth</h1>
  <p>Get an instant, data-driven market price estimate for any used car — just describe the vehicle below.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap="large")

# ───────────────────────── Input form ─────────────────────────
with left:
    st.markdown('<div class="sec-title">🔧 Vehicle details</div>', unsafe_allow_html=True)
    brand = st.selectbox("Brand", CATS["brand"], index=CATS["brand"].index("Toyota") if "Toyota" in CATS["brand"] else 0)

    with st.form("car_form", border=False):
        c1, c2 = st.columns(2)
        if models_by_brand and brand in models_by_brand:
            model_name = c1.selectbox("Model", models_by_brand[brand])
            model_freq = freq_map.get(model_name, np.mean(list(freq_map.values())))
            pop_label = None
        else:
            pop_label = c1.selectbox("Model popularity", list(POPULARITY),
                                     index=2, help="How often this model appears in the market. "
                                     "Upload the dataset in the sidebar for an exact value.")
            model_freq = POPULARITY[pop_label]
        year = c2.number_input("Model year", 1990, REF_YEAR, 2018, step=1)

        c3, c4 = st.columns(2)
        mileage = c3.number_input("Mileage (miles)", 0, 500_000, 45_000, step=1_000)
        fuel = c4.selectbox("Fuel type", CATS["fuel_type"], index=CATS["fuel_type"].index("Gasoline"))

        c5, c6 = st.columns(2)
        hp = c5.number_input("Horsepower (0 = unknown)", 0, 1500, 250, step=10)
        disp = c6.number_input("Engine size in L (0 = unknown)", 0.0, 9.0, 2.5, step=0.1, format="%.1f")

        transmission = st.selectbox("Transmission", CATS["transmission"],
                                    index=CATS["transmission"].index("6-Speed A/T") if "6-Speed A/T" in CATS["transmission"] else 0)

        c7, c8 = st.columns(2)
        ext = c7.selectbox("Exterior colour", CATS["ext_col"],
                           index=CATS["ext_col"].index("Black") if "Black" in CATS["ext_col"] else 0)
        inte = c8.selectbox("Interior colour", CATS["int_col"],
                            index=CATS["int_col"].index("Black") if "Black" in CATS["int_col"] else 0)

        c9, c10 = st.columns(2)
        accident = c9.selectbox("Accident history", CATS["accident"] + ["Unknown"], index=1)
        clean = c10.selectbox("Clean title", ["Yes", "Unknown / Not clean"])

        submitted = st.form_submit_button("🚀  Estimate price")

# ───────────────────────── Result panel ─────────────────────────
with right:
    st.markdown('<div class="sec-title">💰 Estimated market value</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div class="card placeholder">
          <div class="big">🔍</div>
          <h3 style="color:#dfe5ff;margin:.4rem 0">Your estimate will appear here</h3>
          <p>Fill in the vehicle details and hit <b>Estimate price</b>.</p>
        </div>""", unsafe_allow_html=True)
    else:
        row = {
            "brand": brand, "fuel_type": fuel, "transmission": transmission,
            "ext_col": ext, "int_col": inte,
            "accident": None if accident == "Unknown" else accident,
            "clean_title": "Yes" if clean == "Yes" else None,
            "milage": float(mileage), "car_age": float(REF_YEAR - year),
            "horsepower": float(hp) if hp > 0 else np.nan,
            "displacement_L": float(disp) if disp > 0 else np.nan,
            "model_freq": float(model_freq),
        }
        with st.spinner("Crunching the numbers…"):
            price = predict_price(row)
        low, high = price * 0.88, price * 1.12
        age = REF_YEAR - year

        st.markdown(f"""
        <div class="price-card">
          <div class="price-label">Estimated price</div>
          <div class="price-value">${price:,.0f}</div>
          <div class="price-range">Likely range &nbsp;<b>${low:,.0f}</b> – <b>${high:,.0f}</b></div>
        </div>""", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi"><div class="v">{age} yr</div><div class="l">Car age</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="v">{mileage/max(age,1):,.0f}</div><div class="l">Miles / year</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="v">{brand}</div><div class="l">Brand</div></div>', unsafe_allow_html=True)

        gmax = max(150_000, round(high * 1.4, -4))
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=price,
            number={"prefix": "$", "valueformat": ",.0f", "font": {"size": 1, "color": "rgba(0,0,0,0)"}},
            gauge={
                "axis": {"range": [0, gmax], "tickcolor": "#6f7aa6", "tickfont": {"color": "#8f9bc9"}},
                "bar": {"color": "#7c5cff", "thickness": 0.32},
                "bgcolor": "rgba(255,255,255,0.04)", "borderwidth": 0,
                "steps": [{"range": [low, high], "color": "rgba(0,212,255,0.35)"}],
            }))
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e8ecf6"})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("The teal band shows the indicative ±12% range around the estimate.")

# ───────────────────────── Feature importance ─────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">📊 What drives the price?</div>', unsafe_allow_html=True)
imp = pd.DataFrame({"feature": model.feature_name(),
                    "gain": model.feature_importance(importance_type="gain")})
imp["share"] = imp["gain"] / imp["gain"].sum() * 100
imp["label"] = imp["feature"].map(FRIENDLY)
imp = imp.sort_values("share")
bar = go.Figure(go.Bar(
    x=imp["share"], y=imp["label"], orientation="h",
    marker=dict(color=imp["share"], colorscale=[[0, "#3b2ea8"], [.5, "#7c5cff"], [1, "#00d4ff"]]),
    text=[f"{v:.1f}%" for v in imp["share"]], textposition="outside",
    hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
bar.update_layout(height=400, margin=dict(l=10, r=40, t=10, b=10),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  xaxis=dict(showgrid=False, visible=False), yaxis=dict(tickfont=dict(size=13)),
                  font={"color": "#dfe5ff"})
st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="foot">Built with Streamlit · LightGBM · Plotly — estimates are for guidance only.</div>',
            unsafe_allow_html=True)