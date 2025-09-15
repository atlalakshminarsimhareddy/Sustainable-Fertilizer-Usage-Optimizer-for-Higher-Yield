import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("Fertilizer_dataset.csv")  
X = data[['Nitrogen','Phosphorus','Potassium','Soil_pH','Rainfall','Sunlight']]
y = data['Yield']

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
importances = model.feature_importances_

st.set_page_config(page_title="Fertilizer Optimizer", layout="wide")
st.title("🌱 Fertilizer Usage Optimizer for Farmers")
st.markdown("Predict crop yield and get fertilizer recommendations using *sliders and visual graphs*.")

col1, col2, col3 = st.columns(3)

with col1:
    N_level = st.select_slider("🌱 Nitrogen", options=["Low", "Medium", "High"], value="Medium")
    N_map = {"Low":80, "Medium":120, "High":180}
    N = N_map[N_level]

    P_level = st.select_slider("🔵 Phosphorus", options=["Low", "Medium", "High"], value="Medium")
    P_map = {"Low":30, "Medium":60, "High":90}
    P = P_map[P_level]

with col2:
    K_level = st.select_slider("🟤 Potassium", options=["Low", "Medium", "High"], value="Medium")
    K_map = {"Low":30, "Medium":60, "High":100}
    K = K_map[K_level]

    pH_level = st.select_slider("🧪 Soil pH", options=["Acidic", "Neutral", "Alkaline"], value="Neutral")
    pH_map = {"Acidic":5.8, "Neutral":6.5, "Alkaline":7.2}
    soil_pH = pH_map[pH_level]

with col3:
    rainfall_level = st.select_slider("🌧 Rainfall", options=["Low", "Medium", "High"], value="Medium")
    rainfall_map = {"Low":80, "Medium":150, "High":250}
    rainfall = rainfall_map[rainfall_level]

    sunlight_level = st.select_slider("☀ Sunlight Hours", options=["Few", "Moderate", "High"], value="Moderate")
    sunlight_map = {"Few":4.5, "Moderate":7, "High":9}
    sunlight = sunlight_map[sunlight_level]

target_yield = st.slider("🎯 Target Yield (tons/ha)", 5, 30, 15)

input_df = pd.DataFrame({
    'Nitrogen':[N],
    'Phosphorus':[P],
    'Potassium':[K],
    'Soil_pH':[soil_pH],
    'Rainfall':[rainfall],
    'Sunlight':[sunlight]
})
predicted_yield = model.predict(input_df)[0]

if predicted_yield >= target_yield:
    st.success(f"✅ Predicted Yield: {predicted_yield:.2f} tons/ha (meets target)")
else:
    st.warning(f"⚠ Predicted Yield: {predicted_yield:.2f} tons/ha (below target)")

def optimize_fertilizer(target_yield, soil_pH, rainfall, sunlight):
    best_yield = 0
    best_combo = None
    for N_opt in range(50, 201, 20):
        for P_opt in range(20, 101, 20):
            for K_opt in range(20, 121, 20):
                X_input = pd.DataFrame({
                    'Nitrogen':[N_opt],
                    'Phosphorus':[P_opt],
                    'Potassium':[K_opt],
                    'Soil_pH':[soil_pH],
                    'Rainfall':[rainfall],
                    'Sunlight':[sunlight]
                })
                pred_y = model.predict(X_input)[0]
                if pred_y >= target_yield and (best_yield==0 or pred_y<best_yield):
                    best_yield = pred_y
                    best_combo = (N_opt, P_opt, K_opt)
    return best_combo, best_yield

combo, achieved_yield = optimize_fertilizer(target_yield, soil_pH, rainfall, sunlight)
if combo:
    st.info(f"💡 Recommended Fertilizer: 🌱 {combo[0]}kg, 🔵 {combo[1]}kg, 🟤 {combo[2]}kg\nPredicted Yield: {achieved_yield:.2f} tons/ha")
else:
    st.warning("No combination found to achieve the target yield.")

st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    fig1, ax1 = plt.subplots(figsize=(2,2))
    ax1.bar(['Target', 'Predicted'], [target_yield, predicted_yield], color=['orange','green'])
    ax1.set_ylabel("Yield")
    ax1.set_title("Yield")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(2,2))
    colors = ['skyblue' if imp>0.1 else 'lightgray' for imp in importances]
    ax2.bar(X.columns, importances, color=colors)
    ax2.set_ylabel("Importance")
    ax2.set_title("Factors")
    ax2.set_xticks(range(len(X.columns)))
    ax2.set_xticklabels(['N','P','K','pH','Rain','Sun'], rotation=45, ha='right', fontsize=8)
    st.pyplot(fig2)

with col3:
    fig3, ax3 = plt.subplots(figsize=(2,2))
    if combo:
        ax3.bar(['N','P','K'], [combo[0], combo[1], combo[2]], color=['green','blue','brown'])
    ax3.set_ylabel("kg/ha")
    ax3.set_title("Fertilizer")
    st.pyplot(fig3)

with col4:
    if combo:
        safe_limits = {'Nitrogen':150, 'Phosphorus':70, 'Potassium':80}
        N_ratio = combo[0]/safe_limits['Nitrogen']
        P_ratio = combo[1]/safe_limits['Phosphorus']
        K_ratio = combo[2]/safe_limits['Potassium']
        avg_ratio = np.mean([N_ratio, P_ratio, K_ratio])

        if avg_ratio <= 1:
            color = 'green'
            message = "Eco-Friendly ✅"
        elif avg_ratio <= 1.3:
            color = 'orange'
            message = "Caution ⚠"
        else:
            color = 'red'
            message = "Excessive ❌"

        fig4, ax4 = plt.subplots(figsize=(2,0.4))
        ax4.barh([0], [avg_ratio], color=color, height=0.5)
        ax4.set_xlim(0,2)
        ax4.set_yticks([])
        ax4.set_xlabel("Usage Ratio")
        st.pyplot(fig4)
        st.markdown(f"{message}")
