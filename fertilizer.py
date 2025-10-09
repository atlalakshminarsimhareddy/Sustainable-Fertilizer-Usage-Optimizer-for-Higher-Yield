import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Fertilizer Optimizer", layout="centered")
st.title("🌾 Fertilizer Usage Optimizer")
st.markdown("Predict crop yield and optimize fertilizer usage using Machine Learning.")

# --- Load Dataset and Train Model ---
data = pd.read_csv("Fertilizer_dataset.csv")
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Soil_pH', 'Rainfall', 'Sunlight']]
y = data['Yield']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- User Input ---
st.subheader("Enter Field Details:")
N = st.slider("Nitrogen (kg/ha)", 50, 200, 120)
P = st.slider("Phosphorus (kg/ha)", 20, 100, 60)
K = st.slider("Potassium (kg/ha)", 20, 120, 60)
pH = st.slider("Soil pH", 5.5, 7.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 50, 300, 150)
sunlight = st.slider("Sunlight (hrs/day)", 4, 10, 7)

target_yield = st.slider("Target Yield (tons/ha)", 5, 30, 15)

# --- Prediction ---
input_df = pd.DataFrame([[N, P, K, pH, rainfall, sunlight]], 
                        columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Soil_pH', 'Rainfall', 'Sunlight'])
predicted_yield = model.predict(input_df)[0]

st.subheader("Predicted Yield:")
if predicted_yield >= target_yield:
    st.success(f"{predicted_yield:.2f} tons/ha ✅ meets target")
else:
    st.warning(f"{predicted_yield:.2f} tons/ha ⚠ below target")

# --- Optional: Simple Recommendation ---
st.subheader("Suggested Fertilizer Adjustment (Basic):")
if predicted_yield < target_yield:
    st.info("Increase Nitrogen or Potassium slightly to reach target yield.")
else:
    st.info("Current fertilizer levels are sufficient.")

# --- Visualization ---
st.subheader("Yield Comparison:")
fig, ax = plt.subplots()
ax.bar(['Target Yield', 'Predicted Yield'], [target_yield, predicted_yield], color=['orange', 'green'])
ax.set_ylabel("Yield (tons/ha)")
st.pyplot(fig)
