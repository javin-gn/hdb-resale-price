import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE STYLING ---
st.set_page_config(page_title="Singapore HDB Valuation Portal", page_icon="🏢", layout="centered")
st.title("🏢 Singapore HDB Resale Valuation Portal")
st.write("Input a flat's structural configurations and micro-location parameters below for an instant machine learning valuation.")

# --- LOAD TRAINED ML ARTIFACTS ---
@st.cache_resource
def load_pipeline():
    with open('hdb_model_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_pipeline()
model = artifacts['model']
expected_features = artifacts['trained_features']
global_mean = artifacts['global_price_mean']
encoding_maps = artifacts['target_encoding_maps']

# --- FORM ENTRY UI ---
with st.form("valuation_form"):
    st.subheader("1. Structural Dimensions & Timeline")
    col1, col2 = st.columns(2)
    with col1:
        floor_area = st.number_input("Floor Area (Sqm)", min_value=30.0, max_value=170.0, value=90.0, step=1.0)
        storey_midpoint = st.slider("Storey Level (Midpoint)", min_value=2.0, max_value=50.0, value=8.0, step=1.0)
        remaining_lease = st.number_input("Remaining Lease (Years)", min_value=40.0, max_value=99.0, value=85.0, step=0.1)
    with col2:
        flat_type = st.selectbox("Flat Type", options=list(encoding_maps['flat_type'].index))
        flat_model = st.selectbox("Flat Model", options=list(encoding_maps['flat_model'].index))
        lease_commence = st.number_input("Lease Commence Date (Year)", min_value=1960, max_value=2026, value=1995, step=1)

    st.subheader("2. Location Parameters")
    col3, col4 = st.columns(2)
    with col3:
        town = st.selectbox("Town", options=['ANG MO KIO', 'BISHAN', 'YISHUN', 'PUNGGOL', 'TAMPINES', 'JURONG EAST']) # Expand this list as needed
        street_name = st.selectbox("Street Name", options=list(encoding_maps['street_name'].index))
        closest_mrt = st.selectbox("Nearest MRT Station", options=list(encoding_maps['closest_mrt_name'].index))
        closest_school = st.selectbox("Nearest Primary School", options=list(encoding_maps['closest_primary_school_name'].index))
    with col4:
        dist_mrt = st.number_input("Distance to Nearest MRT (Km)", min_value=0.01, max_value=5.0, value=0.45, step=0.01)
        dist_hub = st.number_input("Distance to Nearest Regional Hub (Km)", min_value=0.01, max_value=15.0, value=1.5, step=0.01)
        dist_school = st.number_input("Distance to Nearest Primary School (Km)", min_value=0.01, max_value=3.0, value=0.3, step=0.01)
        dist_mall = st.number_input("Distance to Nearest Shopping Mall (Km)", min_value=0.01, max_value=5.0, value=0.6, step=0.01)

    st.subheader("3. Neighborhood Counts")
    col5, col6, col7 = st.columns(3)
    with col5:
        mrt_1km = st.number_input("MRTs within 1km", min_value=0, max_value=5, value=1)
        mrt_500m = st.number_input("MRTs within 500m", min_value=0, max_value=3, value=0)
    with col6:
        sch_1km = st.number_input("Schools within 1km", min_value=0, max_value=10, value=2)
        sch_2km = st.number_input("Schools within 2km", min_value=0, max_value=20, value=8)
    with col7:
        malls_1km = st.number_input("Malls within 1km", min_value=0, max_value=10, value=1)
        malls_500m = st.number_input("Malls within 500m", min_value=0, max_value=5, value=0)

    # Set constants matching your era filter
    lrt_500m, dist_lrt = 0, 2.5 
    
    submit_btn = st.form_submit_button("Calculate Instant Valuation", type="primary")

# --- STEP 3: PIPELINE EXECUTION FOR PREDICTION ---
if submit_btn:
    # Build single raw observation row mapping inputs
    raw_entry = {
        'floor_area_sqm': floor_area, 'lease_commence_date': lease_commence, 'remaining_lease_years': remaining_lease,
        'storey_midpoint': storey_midpoint, 'year': 2026, 'month_num': 6, 'min_distance_to_regional_hub_km': dist_hub,
        'primary_schools_within_1km': sch_1km, 'primary_schools_within_2km': sch_2km, 'dist_to_closest_primary_school_km': dist_school,
        'mrt_within_500m': mrt_500m, 'mrt_within_1km': mrt_1km, 'dist_to_closest_mrt_km': dist_mrt, 
        'lrt_within_500m': lrt_500m, 'dist_to_closest_lrt_km': dist_lrt, 'malls_within_500m': malls_500m, 'malls_within_1km': malls_1km, 
        'dist_to_closest_shopping_mall_km': dist_mall, 'town': town, 'street_name': street_name, 'closest_mrt_name': closest_mrt,
        'closest_primary_school_name': closest_school, 'flat_model': flat_model, 'flat_type': flat_type
    }
    
    df_sample = pd.DataFrame([raw_entry])
    
    # Process Target Encodings via training maps
    for col, mapping in encoding_maps.items():
        df_sample[f'{col}_encoded'] = df_sample[col].map(mapping).fillna(global_mean)
    df_sample = df_sample.drop(columns=list(encoding_maps.keys()))
    
    # One-Hot Encode Town and align layout structure
    df_sample_encoded = pd.get_dummies(df_sample, columns=['town'], dtype=int)
    X_final_row = df_sample_encoded.reindex(columns=expected_features, fill_value=0)
    
    # Predict Price
    predicted_array = model.predict(X_final_row)
    valuation_result = float(predicted_array[0])
    
    # Display Output
    st.success("Valuation Computed Successfully!")
    st.metric(label="Estimated Resale Price (SGD)", value=f"${valuation_result:,.2f}")
