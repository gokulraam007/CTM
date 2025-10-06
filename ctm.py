import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CTM Loss Calculator", layout="wide")

st.title("🔆 Cell-to-Module (CTM) Loss Calculator")
st.markdown("""
This interactive calculator demonstrates how different design and material parameters 
impact **CTM (Cell-to-Module) efficiency losses**.  
Adjust the sliders and dropdowns to see the effect!
""")

# --- User Inputs ---
st.sidebar.header("Input Parameters")

cell_eff = st.sidebar.slider("Cell Efficiency (%)", 20.0, 25.0, 22.5, 0.1)
num_cells = st.sidebar.selectbox("Number of Cells per Module", [60, 72, 120, 144])
wafer_size = st.sidebar.selectbox("Wafer Size (mm)", ["182", "210"])
connector_type = st.sidebar.selectbox("Connector Type", ["Round Ribbon", "Rectangular Ribbon", "Wire (MBB)"])
encap_type = st.sidebar.selectbox("Encapsulation Material", ["EVA", "POE", "Other"])
finger_coverage = st.sidebar.slider("Finger Coverage (surface %)", 3, 8, 5)
contact_res = st.sidebar.slider("Contact Resistance (mΩ·cm²)", 0.1, 2.0, 0.5, 0.1)

# --- Simple Loss Models ---
# Optical loss: depends on finger coverage and encapsulation type
optical_loss = 0.2 * finger_coverage
if encap_type == "EVA":
    optical_loss += 0.5
elif encap_type == "POE":
    optical_loss += 0.3

# Resistive loss: depends on connector type and contact resistance
if connector_type == "Round Ribbon":
    resistive_loss = 1.0 + contact_res * 0.2
elif connector_type == "Rectangular Ribbon":
    resistive_loss = 0.7 + contact_res * 0.15
else:  # MBB wire
    resistive_loss = 0.5 + contact_res * 0.1

# Mismatch loss: grows with module size
mismatch_loss = 0.5 if num_cells <= 72 else 0.8

# Total CTM loss
total_ctm_loss = optical_loss + resistive_loss + mismatch_loss
module_eff = cell_eff - total_ctm_loss

# --- Output Results ---
st.subheader("📊 Results")
st.metric(label="Cell Efficiency (%)", value=f"{cell_eff:.2f}")
st.metric(label="Module Efficiency (%)", value=f"{module_eff:.2f}")
st.metric(label="Total CTM Loss (%)", value=f"{total_ctm_loss:.2f}")

# Loss breakdown chart
loss_labels = ["Optical", "Resistive", "Mismatch"]
loss_values = [optical_loss, resistive_loss, mismatch_loss]

fig, ax = plt.subplots()
ax.pie(loss_values, labels=loss_labels, autopct='%1.1f%%', startangle=90)
ax.set_title("CTM Loss Breakdown")
st.pyplot(fig)

st.info("""
👉 Note: This is a simplified demo calculator.  
Real CTM analysis uses detailed electrical, optical, and geometrical modeling with lab data.  
""")
