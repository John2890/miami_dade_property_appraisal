import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="Miami-Dade Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

model = load_model()

# --- The 26 columns the model was trained on, in exact order ---
MODEL_COLUMNS = [
    'living_sqft', 'lot_sqft', 'building_age', 'eff_age', 'num_buildings',
    'is_condo', 'Bedroom Count', 'total_baths', 'Floor Count',
    'sqft_per_bedroom', 'bath_bed_ratio',
    'zip_33033', 'zip_33034', 'zip_33035', 'zip_33131', 'zip_33132',
    'zip_33133', 'zip_33139', 'zip_33141', 'zip_33157', 'zip_33160',
    'zip_33178', 'zip_33179', 'zip_33180', 'zip_33186', 'zip_other'
]

# Zip codes the model knows about (the ones that got their own dummy column)
KNOWN_ZIPS = [
    "33033", "33034", "33035", "33131", "33132", "33133",
    "33139", "33141", "33157", "33160", "33178", "33179",
    "33180", "33186"
]

# --- Header ---
st.title("🏠 Miami-Dade Property Price Predictor")
st.markdown(
    "Enter property details below to get an estimated sale price, "
    "powered by a Random Forest model trained on ~45,000 Miami-Dade County "
    "residential sales records."
)
st.divider()

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", ["Single Family", "Condo"])
    living_sqft = st.number_input("Living Area (sq ft)", min_value=100, max_value=15000, value=1500, step=50)
    lot_sqft = st.number_input("Lot Size (sq ft)", min_value=0, max_value=200000, value=5000, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)

with col2:
    zip_code = st.selectbox("Zip Code", KNOWN_ZIPS + ["Other"], index=0)
    total_baths = st.number_input("Total Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    floors = st.number_input("Floors", min_value=1, max_value=50, value=1, step=1)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1990, step=1)

num_buildings = st.number_input("Number of Buildings on Property", min_value=1, max_value=10, value=1, step=1)

# Renovation option
is_renovated = st.checkbox("Property was renovated / updated")
if is_renovated:
    year_renovated = st.number_input(
        "Year Renovated",
        min_value=year_built,
        max_value=2026,
        value=max(year_built, 2010),
        step=1
    )
else:
    year_renovated = year_built

st.divider()

# --- Predict ---
if st.button("💰 Predict Sale Price", use_container_width=True):

    # Calculate derived features
    current_year = datetime.now().year
    building_age = current_year - year_built
    eff_age = current_year - year_renovated  # uses renovation year if provided
    is_condo = 1 if property_type == "Condo" else 0

    # Guard against division by zero
    sqft_per_bedroom = living_sqft / bedrooms if bedrooms > 0 else living_sqft
    bath_bed_ratio = total_baths / bedrooms if bedrooms > 0 else total_baths

    # Build a single-row DataFrame with all 26 columns, initialized to 0
    input_data = pd.DataFrame(np.zeros((1, len(MODEL_COLUMNS))), columns=MODEL_COLUMNS)

    # Fill in the base features
    input_data['living_sqft'] = living_sqft
    input_data['lot_sqft'] = lot_sqft
    input_data['building_age'] = building_age
    input_data['eff_age'] = eff_age
    input_data['num_buildings'] = num_buildings
    input_data['is_condo'] = is_condo
    input_data['Bedroom Count'] = bedrooms
    input_data['total_baths'] = total_baths
    input_data['Floor Count'] = floors
    input_data['sqft_per_bedroom'] = sqft_per_bedroom
    input_data['bath_bed_ratio'] = bath_bed_ratio

    # Set the correct zip code dummy to 1
    if zip_code == "Other":
        input_data['zip_other'] = 1
    else:
        col_name = f"zip_{zip_code}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"### Estimated Sale Price: ${prediction:,.0f}")

    # Context info
    reno_note = f", renovated in {year_renovated}" if is_renovated else ""
    st.caption(
        f"Based on a {living_sqft:,} sq ft {'condo' if is_condo else 'single-family home'} "
        f"with {bedrooms} bed / {total_baths:.1f} bath, built in {year_built}{reno_note}, "
        f"in zip code {zip_code}."
    )

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Random Forest model · R² = 0.78 · MAE ≈ $96K · "
    "Trained on Florida DOR + Miami-Dade Open Data Hub records"
    "</div>",
    unsafe_allow_html=True
)
