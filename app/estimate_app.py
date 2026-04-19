import pickle
import joblib
from pathlib import Path

import streamlit as st
import pandas as pd

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Warsaw Apartment Price Estimator",
    page_icon="\U0001f3e0",
    layout="centered",
)

# -- Model loading -------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def get_model():
    return joblib.load(MODELS_DIR / "price_model.pkl")


@st.cache_resource
def get_feature_columns():
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as fh:
        return pickle.load(fh)


# -- District encoding ---------------------------------------------------------
# LabelEncoder used during data preparation encodes alphabetically (0-indexed).
DISTRICT_MAP = {
    "Bemowo": 0,
    "Bia\u0142o\u0142\u0119ka": 1,
    "Bielany": 2,
    "Mokot\u00f3w": 3,
    "Ochota": 4,
    "Praga Po\u0142udnie": 5,
    "Praga P\u00f3\u0142noc": 6,
    "Rembertow": 7,
    "Targ\u00f3wek": 8,
    "Ursus": 9,
    "Ursynow": 10,
    "Wawer": 11,
    "Weso\u0142a": 12,
    "Wilanow": 13,
    "Wola": 14,
    "W\u0142ochy": 15,
    "\u015ar\u00f3dmie\u015bcie": 16,
    "\u017boliborz": 17,
}

# river_side: 0 = east bank (right side), 1 = west bank (left side)
EAST_BANK = {
    "Bia\u0142o\u0142\u0119ka",
    "Praga Po\u0142udnie",
    "Praga P\u00f3\u0142noc",
    "Rembertow",
    "Targ\u00f3wek",
    "Wawer",
    "Weso\u0142a",
}


def river_side_for(district_name):
    return 0 if district_name in EAST_BANK else 1


# -- UI ------------------------------------------------------------------------
st.title("\U0001f3e0 Warsaw Apartment Price Estimator")
st.caption(
    "Predict the price per m\u00b2 (PLN) using a Random Forest model "
    "trained on Warsaw real-estate listings."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property details")
    rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3, step=1)
    construction_year = st.number_input(
        "Year of construction", min_value=1900, max_value=2030, value=2000, step=1
    )
    current_year = pd.Timestamp.now().year
    building_age = current_year - construction_year

with col2:
    st.subheader("Location")
    district_name = st.selectbox("District", options=sorted(DISTRICT_MAP.keys()))
    dist_to_metro_km = st.number_input(
        "Distance to metro (km)", min_value=0.0, max_value=50.0, value=1.0, step=0.1, format="%.2f"
    )
    dist_to_centrum_km = st.number_input(
        "Distance to city centre (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1, format="%.2f"
    )

st.divider()

# -- Derived features ----------------------------------------------------------
district_code = DISTRICT_MAP[district_name]
river_side_code = river_side_for(district_name)
bank_label = (
    "East bank (right side of Vistula)"
    if river_side_code == 0
    else "West bank (left side of Vistula)"
)

with st.expander("Derived values used by the model", expanded=False):
    st.markdown(
        f"""
        | Feature | Value |
        |---|---|
        | Building age | **{building_age}** years |
        | District code | **{district_code}** |
        | River side | **{river_side_code}** ({bank_label}) |
        """
    )

# -- Prediction ----------------------------------------------------------------
if st.button("Estimate price per m\u00b2", type="primary", use_container_width=True):
    model = get_model()
    feature_columns = get_feature_columns()

    input_dict = {
        "building_age": building_age,
        "district": district_code,
        "dist_to_centrum_km": dist_to_centrum_km,
        "river_side": river_side_code,
        "rooms": rooms,
        "dist_to_metro_km": dist_to_metro_km,
    }

    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns)

    try:
        price_per_m2 = model.predict(input_df)[0]
        st.success(f"### Estimated price: **{price_per_m2:,.0f} PLN / m\u00b2**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
