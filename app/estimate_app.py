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

# -- Model + artifact loading --------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def get_model():
    return joblib.load(MODELS_DIR / "price_model.pkl")


@st.cache_resource
def get_feature_columns():
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as fh:
        return pickle.load(fh)


@st.cache_resource
def get_district_tier_map():
    """dict: district_name -> tier (int 0-6, data-driven from training set)."""
    with open(MODELS_DIR / "district_tier_map.pkl", "rb") as fh:
        return pickle.load(fh)


# -- Derive available material groups from saved feature columns ---------------
@st.cache_resource
def get_material_groups():
    feature_cols = get_feature_columns()
    return sorted([c.replace("material_group_", "") for c in feature_cols if c.startswith("material_group_")])


# -- River-side lookup (1 = east/right bank, 0 = west/left bank) ---------------
# Matches the encoding in 03_modify.ipynb
EAST_BANK = {
    "Bia\u0142o\u0142\u0119ka",
    "Praga Po\u0142udnie",
    "Praga P\u00f3\u0142noc",
    "Rembertow",
    "Targ\u00f3wek",
    "Wawer",
    "Weso\u0142a",
}


def river_side_for(district_name: str) -> int:
    return 1 if district_name in EAST_BANK else 0


# -- UI ------------------------------------------------------------------------
st.title("\U0001f3e0 Warsaw Apartment Price Estimator")
st.caption(
    "Predict the price per m\u00b2 (PLN) using a Random Forest model "
    "trained on Warsaw real-estate listings."
)
st.divider()

district_tier_map = get_district_tier_map()
material_groups   = get_material_groups()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property details")
    rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3, step=1)
    surface = st.number_input("Surface area (m\u00b2)", min_value=10.0, max_value=500.0, value=60.0, step=1.0)
    floor = st.number_input("Floor number", min_value=-1, max_value=30, value=2, step=1)
    construction_year = st.number_input(
        "Year of construction", min_value=1900, max_value=2030, value=2000, step=1
    )
    market = st.selectbox("Market type", options=[(0, "Primary"), (1, "Secondary")], format_func=lambda x: x[1])[0]
    material_group = st.selectbox("Building material", options=material_groups)

with col2:
    st.subheader("Location")
    district_name = st.selectbox("District", options=sorted(district_tier_map.keys()))
    dist_to_metro = st.number_input(
        "Distance to nearest metro station (km)", min_value=0.0, max_value=30.0, value=1.0, step=0.1, format="%.2f"
    )
    dist_to_center = st.number_input(
        "Distance to city centre (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1, format="%.2f"
    )

st.divider()

# -- Derived features ----------------------------------------------------------
current_year = pd.Timestamp.now().year
building_age     = current_year - construction_year
district_tier    = int(district_tier_map[district_name])
river_side_code  = river_side_for(district_name)
bank_label = "East bank (right side of Vistula)" if river_side_code == 1 else "West bank (left side of Vistula)"

with st.expander("Derived values used by the model", expanded=False):
    st.markdown(
        f"""
        | Feature | Value |
        |---|---|
        | Building age | **{building_age}** years |
        | District price tier | **{district_tier}** / 6 (0 = cheapest area, 6 = most expensive) |
        | River side | **{river_side_code}** ({bank_label}) |
        """
    )

# -- Prediction ----------------------------------------------------------------
if st.button("Estimate price per m\u00b2", type="primary", use_container_width=True):
    model = get_model()
    feature_columns = get_feature_columns()

    # Build input row: start with zeros for all features
    input_dict = {col: 0 for col in feature_columns}

    # Set OHE material group dummy
    mat_col = f"material_group_{material_group}"
    if mat_col in input_dict:
        input_dict[mat_col] = 1

    # Set numeric features
    input_dict["building_age"]   = building_age
    input_dict["dist_to_center"] = dist_to_center
    input_dict["dist_to_metro"]  = dist_to_metro
    input_dict["floor"]          = floor
    input_dict["market"]         = market
    input_dict["river_side"]     = river_side_code
    input_dict["rooms"]          = rooms
    input_dict["surface"]        = surface
    input_dict["district_tier"]  = district_tier

    input_df = pd.DataFrame([input_dict])[feature_columns]

    try:
        price_per_m2 = model.predict(input_df)[0]
        st.success(f"### Estimated price: **{price_per_m2:,.0f} PLN / m\u00b2**")
        if surface > 0:
            total = price_per_m2 * surface
            st.info(f"Total estimated value for {surface:.0f} m\u00b2: **{total:,.0f} PLN**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
