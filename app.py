import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)


model = pickle.load(open('housing.pkl', 'rb'))
columns= pickle.load(open('columns.pkl', 'rb'))


st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.big-card {
    background: linear-gradient(135deg,#6c63ff,#4da3ff);
    padding: 40px;
    border-radius: 25px;
    color: white;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
}

.small-card {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.05);
    text-align: center;
}

.label {
    color: gray;
    font-size: 14px;
}

.value {
    font-size: 28px;
    font-weight: bold;
    color: #111;
}

.sidebar .sidebar-content {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)


st.sidebar.title("🏠 Home Details")

location = st.sidebar.selectbox(
    "Choose Location",
    [
        "<1H OCEAN",
        "INLAND",
        "NEAR BAY",
        "NEAR OCEAN",
        "ISLAND"
    ]
)

house_age = st.sidebar.number_input(
    "House Age (Years)",
    min_value=1,
    max_value=52,
    value=25
)

rooms = st.sidebar.number_input(
    "Total Rooms",
    min_value=2,
    max_value=40000,
    value=2000
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=7000,
    value=400
)

population = st.sidebar.number_input(
    "Population in Area",
    min_value=3,
    max_value=35000,
    value=1500
)

households = st.sidebar.number_input(
    "Households",
    min_value=1,
    max_value=6000,
    value=500
)

income = st.sidebar.slider(
    "Median Income",
    0.0,
    15.0,
    3.5
)

longitude = st.sidebar.slider(
    "Longitude",
    -125.0,
    -113.0,
    -120.0
)

latitude = st.sidebar.slider(
    "Latitude",
    32.0,
    42.0,
    37.0
)

rooms_log = np.log(rooms)
bedrooms_log = np.log(bedrooms)
population_log = np.log(population)
households_log = np.log(households)

input_data = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": house_age,
    "total_rooms": rooms_log,
    "total_bedrooms": bedrooms_log,
    "population": population_log,
    "households": households_log,
    "median_income": income
}

ocean_columns = [
    "ocean_proximity_<1H OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN"
]

for col in ocean_columns:
    input_data[col] = 0

selected_col = f"ocean_proximity_{location}"
input_data[selected_col] = 1

input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(
    columns=columns,
    fill_value=0
)


prediction = model.predict(input_df)[0]
st.title("🏠 California Housing Price Predictor")

st.write(
    "Get an estimated house price based on location and property details."
)

st.markdown(f"""
<div class="big-card">

<h2>Estimated House Price</h2>

<h1 style="font-size:60px;">
${prediction:,.0f}
</h1>

<p>
This estimate is based on:
</p>

<ul>
<li>Location & neighborhood</li>
<li>Property details</li>
<li>Market conditions</li>
</ul>

</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="small-card">
    <div class="label">📍 Location</div>
    <div class="value">{location}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="small-card">
    <div class="label">🏠 House Age</div>
    <div class="value">{house_age} yrs</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="small-card">
    <div class="label">🛏 Rooms</div>
    <div class="value">{rooms}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="small-card">
    <div class="label">💰 Income</div>
    <div class="value">{income}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.info(
    "This prediction is generated using Machine Learning based on California housing data."
)
