# Gaming addiction risk app.  Run with:  streamlit run app.py
# Needs gaming_addiction_classifier.pkl (from model.py) and data/gaming_addiction.csv

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gaming Addiction Risk", page_icon="🎮")


@st.cache_resource
def load_model():
    return joblib.load("gaming_addiction_classifier.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/gaming_addiction.csv", keep_default_na=False, na_values="")


model = load_model()
data = load_data()
features = list(model.feature_names_in_)

# The model needs all 42 columns, so we ask for the most important ones
# and fill the rest with a typical value (median for numbers, most common for text).
typical = {}
for col in features:
    if pd.api.types.is_numeric_dtype(data[col]):
        typical[col] = data[col].median()
    else:
        typical[col] = data[col].mode()[0]

st.title("🎮 Gaming Addiction Risk Estimator")
st.write("Move the sliders to describe a player. All other details are set to typical values.")

player = pd.DataFrame([typical])

sliders = {
    "self_control_score": "Self-control score",
    "impulsiveness_score": "Impulsiveness score",
    "dopamine_dependency_index": "Dopamine dependency index",
    "daily_playtime_hours": "Daily playtime (hours)",
    "late_night_sessions_hours": "Late-night gaming (hours)",
    "sleep_hours": "Sleep (hours)",
    "absenteeism_days": "Absenteeism (days)",
}

for col, label in sliders.items():
    low, high = float(data[col].min()), float(data[col].max())
    player[col] = st.slider(label, low, high, float(typical[col]))

player["platform"] = st.selectbox("Platform", sorted(data["platform"].unique()),
                                  index=sorted(data["platform"].unique()).index(typical["platform"]))

probability = model.predict_proba(player[features])[0, 1]

st.subheader("Result")
st.metric("Estimated chance of addiction", f"{probability:.0%}")
st.progress(float(probability))

if probability >= 0.5:
    st.error("The model classes this player as AT RISK.")
else:
    st.success("The model classes this player as NOT at risk.")

st.caption("Educational demo trained on a small public Kaggle dataset (250 players). "
           "Not a medical or diagnostic tool.")
