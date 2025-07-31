import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('wine_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🍷 Wine Quality Checker",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- PAGE STYLE ---
st.markdown("""
    <style>
    body {
        background-color: #f2f0eb;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #800000;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #a00000;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🍷 Wine Quality Predictor")
st.markdown("**Predict if a wine is of good or bad quality based on its chemical properties.**")

# --- INPUTS ---
st.subheader("🔬 Enter Wine Characteristics")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, value=7.4, step=0.1)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, value=0.36, step=0.01)
    chlorides = st.number_input("Chlorides", 0.01, 0.2, value=0.076, step=0.001)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6, 300, value=34)
    pH = st.number_input("pH", 2.5, 4.5, value=3.51, step=0.01)
    alcohol = st.number_input("Alcohol (%)", 8.0, 15.0, value=9.4, step=0.1)

with col2:
    volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, value=0.7, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, value=1.9, step=0.1)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1, 75, value=11)
    density = st.number_input("Density", 0.9900, 1.0050, value=0.9978, step=0.0001, format="%.4f")
    sulphates = st.number_input("Sulphates", 0.2, 2.0, value=0.56, step=0.01)

# --- PREDICT BUTTON ---
if st.button("🔍 Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                          density, pH, sulphates, alcohol]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    label = "🍇 Good Quality Wine" if pred == 1 else "⚠️ Bad Quality Wine"

    st.success(f"**Prediction:** {label}")

# --- FOOTER ---
st.markdown("</div>", unsafe_allow_html=True)
