import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import json

# Page config
st.set_page_config(page_title="Planetary Defense Analytics", layout="wide")

st.title("Planetary Defense Analytics")
st.write("Enter asteroid parameters to predict PHA hazard, diameter, and cluster.")

# ========== Load Models ==========
@st.cache_resource
def load_models():
    pha_model = joblib.load('models/pha_model.pkl')
    diameter_model = joblib.load('models/diameter_model.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    scaler_reg = joblib.load('models/scaler_reg.pkl')
    scaler_cluster = joblib.load('models/scaler_cluster.pkl')

    with open('data/processed/selected_features.json', 'r') as f:
        features = json.load(f)

    return (pha_model, diameter_model, kmeans_model,
            scaler, scaler_reg, scaler_cluster, features)

(pha_model, diameter_model, kmeans_model,
 scaler, scaler_reg, scaler_cluster, selected_features) = load_models()

# ========== Input Section ==========
st.header("Asteroid Parameters")

col1, col2 = st.columns(2)

with col1:
    a = st.number_input("Semi-major axis (a) [AU]", 0.1, 100.0, 2.5)
    e = st.number_input("Eccentricity (e)", 0.0, 1.0, 0.3)
    i = st.number_input("Inclination (i) [deg]", 0.0, 180.0, 15.0)
    om = st.number_input("Longitude of node (om) [deg]", 0.0, 360.0, 100.0)

with col2:
    w = st.number_input("Argument of perihelion (w) [deg]", 0.0, 360.0, 50.0)
    ma = st.number_input("Mean anomaly (ma) [deg]", 0.0, 360.0, 200.0)
    q = st.number_input("Perihelion (q) [AU]", 0.0, 50.0, 1.8)
    n = st.number_input("Mean motion (n)", 0.0, 5.0, 0.25)


# ========== PREDICT Button ==========
if st.button("PREDICT"):

    # Build input for Case 1 & 2 (selected features)
    input_data = pd.DataFrame({
        'n_log': [np.log1p(abs(n))],
        'w': [w],
        'ma': [ma],
        'e_log': [np.log1p(abs(e))],
        'a_log': [np.log1p(abs(a))],
        'q_log': [np.log1p(abs(q))]
    })
    input_data = input_data[selected_features]

    # ============ MODULE 1: PHA ============
    st.markdown("---")
    st.header("Module 1: PHA Hazard Classification")

    pha_scaled = scaler.transform(input_data)
    pha_pred = pha_model.predict(pha_scaled)[0]
    pha_prob = pha_model.predict_proba(pha_scaled)[0][1]

    if pha_pred == 1:
        st.error(f"POTENTIALLY HAZARDOUS — Confidence: {pha_prob*100:.1f}%")
    else:
        st.success(f"NOT Hazardous — Confidence: {(1-pha_prob)*100:.1f}%")

    # SHAP for PHA
    st.subheader("SHAP Explanation")
    try:
        explainer = shap.TreeExplainer(pha_model)
        shap_vals = explainer.shap_values(pha_scaled)

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
            base = explainer.expected_value[1]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
            base = explainer.expected_value[1]
        elif len(shap_vals.shape) == 2:
            sv = shap_vals[:, 1]
            base = explainer.expected_value[1]
        else:
            sv = shap_vals[0]
            base = explainer.expected_value

        fig, ax = plt.subplots(figsize=(6, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=base,
                data=pha_scaled[0],
                feature_names=selected_features
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as ex:
        st.warning(f"SHAP plot error: {ex}")

    # ============ MODULE 2: Diameter ============
    st.markdown("---")
    st.header("Module 2: Diameter Prediction")

    reg_scaled = scaler_reg.transform(input_data)
    diameter_pred = diameter_model.predict(reg_scaled)[0]

    st.info(f"Predicted Diameter: {diameter_pred:.2f} km")

    # SHAP for Diameter
    st.subheader("SHAP Explanation")
    try:
        try:
            explainer_reg = shap.TreeExplainer(diameter_model)
            shap_vals_reg = explainer_reg.shap_values(reg_scaled)
            base_reg = explainer_reg.expected_value
        except Exception:
            explainer_reg = shap.LinearExplainer(diameter_model, reg_scaled)
            shap_vals_reg = explainer_reg.shap_values(reg_scaled)
            base_reg = explainer_reg.expected_value

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals_reg[0],
                base_values=base_reg,
                data=reg_scaled[0],
                feature_names=selected_features
            ),
            show=False
        )
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception as ex:
        st.warning(f"SHAP plot error: {ex}")

    # ============ MODULE 3: Cluster ============
    st.markdown("---")
    st.header("Module 3: Cluster Assignment")

    cluster_data = pd.DataFrame({
        'a_log': [np.log1p(abs(a))],
        'e_log': [np.log1p(abs(e))],
        'i_log': [np.log1p(abs(i))],
        'q_log': [np.log1p(abs(q))],
        'n_log': [np.log1p(abs(n))],
        'om': [om],
        'w': [w],
        'ma': [ma]
    })

    cluster_scaled = scaler_cluster.transform(cluster_data)
    cluster = kmeans_model.predict(cluster_scaled)[0]
    st.info(f"Assigned Cluster: Cluster {cluster}")

