"""
CTM Technical Calculator - Streamlit app
Author: (You can add your name)
Purpose: Technical interactive CTM (Cell-to-Module) loss estimator for R&D use.
Notes:
 - This model combines physics-based approximations with tunable empirical coefficients.
 - Calibrate coefficients (see COEFFICIENTS block) against lab measurements or literature.
 - Units: lengths in mm (unless noted), resistances in Ohm or mOhm·cm^2 where indicated.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="CTM Technical Calculator", layout="wide")

# === CONFIG / TUNABLE COEFFICIENTS ===
# Tune these values based on lab calibration or literature references
COEFF = {
    # Optical: shadowing effectiveness (how much finger coverage blocks current) [fraction of finger coverage]
    "shadow_factor": 0.90,  # ~90% of finger geometric coverage translates to current shading
    # Optical: encapsulation scattering/reflection penalty (absolute %) per encapsulant type (empirical)
    "encap_penalty": {"EVA": 0.4, "POE": 0.25, "Other": 0.45},  # in percentage points
    # Resistive: baseline scaling to convert contact resistance (mOhm·cm2) to module-level efficiency loss %
    "contact_to_pct": 1.8,  # % loss at 0.5 mΩ·cm² roughly -> scale accordingly in computation below
    # Resistive: finger resistance scaling factor (depends on finger geometry & sheet resistance)
    "finger_scale": 0.12,  # multiplier mapping finger area/resistance to % loss (empirical)
    # Busbar effect factor (more busbars reduces resistive loss)
    "busbar_factor_per_busbar": 0.85,  # multiplicative factor per additional busbar after 1
    # Mismatch: base mismatch per cell-to-cell relative variation (std dev in %)
    "mismatch_base": 0.20,   # % per 1% cell variation baseline for modules of ~60 cells (scales with size)
    # Misc: baseline irradiance conversion for energy yield (kWh estimate)
    "baseline_irradiance_kwh_per_m2_per_year": 1400  # placeholder; change per site
}

# === Helper functions (models & calculations) ===

def compute_optical_loss(finger_coverage_pct, inactive_area_pct, encap_type, reflection_pct):
    """
    Optical loss model:
      - Shadowing by metallization (fingers & busbars) converted by shadow_factor
      - Inactive area (frame/pads) directly reduces active area
      - Encapsulation adds a penalty (index mismatch, scattering)
      - Reflection adds a small additional loss
    Returns loss in percentage points (e.g., 1.25 means 1.25% absolute loss)
    """
    shadow_loss = (finger_coverage_pct / 100.0) * COEFF["shadow_factor"] * 100.0  # in %
    inactive_loss = inactive_area_pct  # already percent of area
    encap_loss = COEFF["encap_penalty"].get(encap_type, COEFF["encap_penalty"]["Other"])
    refl_loss = reflection_pct  # user-provided reflection %
    optical_loss = shadow_loss + inactive_loss + encap_loss + refl_loss
    return optical_loss

def compute_resistive_loss(contact_res_mohm_cm2, finger_area_mm2, total_finger_length_mm,
                           finger_width_mm, busbar_res_ohm_per_m, num_busbars, connector_type,
                           string_length_cells, series_connection=True):
    """
    Resistive loss model: combine multiple contributions into approximate % loss.
    - Contact resistance contribution scaled by contact_to_pct coefficient (empirical).
    - Finger resistance proxy: depends on finger area and length; less area or longer fingers -> more loss.
    - Busbar contribution: based on linear resistance and current path length.
    - Interconnection/connector type factor modifies the resistive loss (wire vs ribbon etc.)
    Returns resistive loss in percentage points.
    """
    # Contact resistance contribution
    # scale mOhm·cm2 -> % using COEFF["contact_to_pct"], normalized to 0.5 mOhm·cm2 baseline
    contact_norm = contact_res_mohm_cm2 / 0.5
    contact_loss = COEFF["contact_to_pct"] * contact_norm  # %

    # Finger resistance proxy: smaller area -> higher loss.
    # finger_area_mm2 is total finger metallization area per cell (or per unit)
    # total_finger_length_mm approximates conduction path length
    if finger_area_mm2 <= 0:
        finger_loss = 0.0
    else:
        finger_loss = COEFF["finger_scale"] * (total_finger_length_mm / (finger_area_mm2 + 1e-9)) * 100.0
        # normalizing factor to bring into % range; empirical

    # Busbar loss approximation
    # busbar_res_ohm_per_m * path_length_m * I_effective^2 -> convert to %
    # We don't have absolute current here; we estimate relative effect using busbar_res value and cell count
    # Use string_length_cells as proxy for path length; assume per busbar path scales roughly linearly
    busbar_path_factor = max(1.0, string_length_cells / 60.0)  # longer modules slightly worse
    busbar_loss = (busbar_res_ohm_per_m * busbar_path_factor * 100.0)  # scaled to percentage proxy

    # Busbar count effect: more busbars reduce effective loss multiplicatively
    if num_busbars <= 1:
        busbar_multiplier = 1.0
    else:
        busbar_multiplier = COEFF["busbar_factor_per_busbar"] ** (num_busbars - 1)

    busbar_loss *= busbar_multiplier

    # Connector type effect
    connector_map = {"Round Ribbon": 1.05, "Rectangular Ribbon": 0.92, "Wire (MBB)": 0.85, "Other": 1.0}
    connector_factor = connector_map.get(connector_type, 1.0)

    resistive_loss = (contact_loss + finger_loss + busbar_loss) * connector_factor

    # Scale down to reasonable engineering percentages:
    # Many raw proxies above are exaggerated; bring into realistic range with an empirical scaling
    resistive_loss = resistive_loss * 0.02  # empirical scaling (tune this)
    resistive_loss = max(0.0, resistive_loss)

    return resistive_loss

def compute_mismatch_loss(cell_std_percent, num_cells):
    """
    Mismatch loss model:
      - Mismatch grows with cell-to-cell variation and module size.
      - Basic approx: mismatch_loss (%) = mismatch_base * cell_std_percent * sqrt(num_cells / 60)
    """
    if cell_std_percent <= 0:
        return 0.0
    mismatch_loss = COEFF["mismatch_base"] * cell_std_percent * np.sqrt(num_cells / 60.0)
    return mismatch_loss

def compute_total_ctm(cell_eff_percent, optical_loss_pct, resistive_loss_pct, mismatch_loss_pct):
    """
    Combine losses. We assume additive percentage-point losses for CTM approximation:
      module_eff = cell_eff - (optical + resistive + mismatch + other small losses)
    Optionally enforce module_eff >= 0.
    """
    total_loss = optical_loss_pct + resistive_loss_pct + mismatch_loss_pct
    module_eff = cell_eff_percent - total_loss
    module_eff = max(0.0, module_eff)
    return module_eff, total_loss

# === Streamlit UI ===

st.title("🔧 CTM Technical Calculator — R&D Edition")
st.markdown("""
This calculator estimates Cell-to-Module (CTM) losses using physics-based proxies and tunable empirical coefficients.
**Intended for R&D engineers** — please calibrate the coefficients with lab data for production accuracy.
""")

# Two-column layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.header("Cell & Module Inputs")
    col1, col2 = st.columns(2)

    with col1:
        cell_eff = st.number_input("Cell Efficiency (%)", min_value=10.0, max_value=30.0, value=22.5, step=0.01,
                                   help="Reference cell efficiency (STC, cell-level).")
        num_cells = st.selectbox("Cells per Module", options=[60, 66, 72, 96, 120, 144], index=2)
        wafer_x = st.selectbox("Wafer dimension X (mm)", options=[182.2, 183.75, 210.0], index=0)
        wafer_y = st.selectbox("Wafer dimension Y (mm)", options=[182.2, 210.0], index=0)

        # Advanced electrical inputs
        Isc_gain_percent = st.number_input("Isc gain due to encapsulation (%)", min_value=-5.0, max_value=10.0, value=0.3,
                                           help="Change in short-circuit current induced by encapsulation / optics.")

    with col2:
        st.subheader("Metallization & Interconnection")
        total_busbar_area = st.number_input("Total busbar area (mm² per cell)", value=40.0, step=0.1,
                                           help="Sum of busbar areas per cell (exclude fingers).")
        total_finger_area = st.number_input("Total finger area (mm² per cell)", value=15.0, step=0.1,
                                           help="Total width*length of fingers per cell (approx).")
        effective_finger_width_encap = st.slider("Effective finger width under encapsulation (%)", 50, 100, 88,
                                                 help="Effective conduction width of fingers after encapsulation.")
        effective_finger_width_air = st.slider("Effective finger width in air (%)", 50, 100, 88)

        num_busbars = st.selectbox("Number of Busbars (per cell)", options=[1,2,3,4,5,6], index=3)
        connector_type = st.selectbox("Interconnection Type", options=["Round Ribbon", "Rectangular Ribbon", "Wire (MBB)", "Other"])

with right_col:
    st.header("Contact & Resistances")
    contact_res_mohm_cm2 = st.slider("Solder / contact resistance (mΩ·cm²)", min_value=0.05, max_value=5.0, value=0.5, step=0.01,
                                     help="Measured resistance of the soldered connection between ribbon and busbar (milliOhm·cm²).")
    effective_busbar_res = st.number_input("Effective busbar resistance (Ω/m)", value=0.0035, step=0.0001,
                                           help="Linear resistance of busbar material per meter.")
    pad_count = st.number_input("Number of pads per row (pcs)", value=3, min_value=1)
    pad_size_mm2 = st.number_input("Pad size (mm²)", value=4.0, step=0.1)

    st.markdown("---")
    st.subheader("Optical / Others")
    finger_coverage_pct = st.slider("Finger coverage (%)", 0.0, 10.0, 4.5, step=0.1,
                                   help="Geometric fraction of front surface covered by fingers (percent).")
    inactive_area_pct = st.slider("Inactive area (%) (frame, gaps, pads)", 0.0, 10.0, 1.2, step=0.1)
    reflection_pct = st.slider("Front-surface reflection (%)", 0.0, 5.0, 1.0, step=0.1,
                               help="Fraction of light reflected from top surface at normal incidence.")

    st.markdown("---")
    st.subheader("Mismatch / Variability")
    cell_std_percent = st.slider("Cell-to-cell current std dev (%)", 0.0, 5.0, 0.6, step=0.01,
                                 help="Relative standard deviation of cell short-circuit current across the module (process variation).")

# Advanced / tuning section (collapsed)
with st.expander("Advanced: Model coefficients and calibration (tweak for lab fit)"):
    st.write("These coefficients are for R&D tuning. Change carefully and re-check results.")
    C1 = st.number_input("shadow_factor (0-1)", value=COEFF["shadow_factor"], format="%.3f")
    C2_eva = st.number_input("encap_penalty (EVA) %", value=COEFF["encap_penalty"]["EVA"], format="%.3f")
    C2_poe = st.number_input("encap_penalty (POE) %", value=COEFF["encap_penalty"]["POE"], format="%.3f")
    C_contact_to_pct = st.number_input("contact_to_pct (scale)", value=COEFF["contact_to_pct"], format="%.3f")
    C_finger_scale = st.number_input("finger_scale (empirical)", value=COEFF["finger_scale"], format="%.4f")
    C_busbar_factor = st.number_input("busbar_factor_per_busbar", value=COEFF["busbar_factor_per_busbar"], format="%.3f")
    C_mismatch_base = st.number_input("mismatch_base (%)", value=COEFF["mismatch_base"], format="%.4f")

    # Apply edits locally (temporary)
    COEFF["shadow_factor"] = float(C1)
    COEFF["encap_penalty"]["EVA"] = float(C2_eva)
    COEFF["encap_penalty"]["POE"] = float(C2_poe)
    COEFF["contact_to_pct"] = float(C_contact_to_pct)
    COEFF["finger_scale"] = float(C_finger_scale)
    COEFF["busbar_factor_per_busbar"] = float(C_busbar_factor)
    COEFF["mismatch_base"] = float(C_mismatch_base)

# === Calculations ===

# Estimate total finger length (proxy) from wafer dimensions and finger pitch assumptions:
# Assume finger pitch such that total_finger_length scales with wafer width and finger count estimate
cell_width_mm = wafer_x  # approximate cell width dimension
estimated_finger_count = max(10, int((cell_width_mm / 0.25)))  # crude approx: many narrow fingers -> higher count
total_finger_length_mm = estimated_finger_count * (wafer_y / 2.0)  # half-cell path average (proxy)
finger_area_mm2 = total_finger_area  # user-provided total finger area per cell proxy

optical_loss = compute_optical_loss(finger_coverage_pct, inactive_area_pct, "EVA" if st.session_state.get('encap','EVA') is None else "EVA", reflection_pct)
# Note above: user encapsulant choice not explicit — use EVA for now or expand UI to choose encapsulant
# Let's add an encapsulant selector:
encap_type = st.session_state.get('encap_type') if 'encap_type' in st.session_state else None
if encap_type is None:
    # set from default earlier (we didn't capture earlier) - add small widget
    encap_type = st.selectbox("Encapsulation (choose for optics model)", options=["EVA","POE","Other"], index=0)
    st.session_state['encap_type'] = encap_type
else:
    encap_type = st.selectbox("Encapsulation (choose for optics model)", options=["EVA","POE","Other"], index=["EVA","POE","Other"].index(encap_type))
    st.session_state['encap_type'] = encap_type

# Recompute optical with chosen encap
optical_loss = compute_optical_loss(finger_coverage_pct, inactive_area_pct, encap_type, reflection_pct)

# Resistive
resistive_loss = compute_resistive_loss(contact_res_mohm_cm2, finger_area_mm2, total_finger_length_mm,
                                       finger_width_mm=(total_finger_area / (estimated_finger_count + 1.0)),
                                       busbar_res_ohm_per_m=effective_busbar_res,
                                       num_busbars=num_busbars,
                                       connector_type=connector_type,
                                       string_length_cells=num_cells)

# Mismatch
mismatch_loss = compute_mismatch_loss(cell_std_percent, num_cells)

module_eff, total_ctm_loss = compute_total_ctm(cell_eff, optical_loss, resistive_loss, mismatch_loss)

# Energy yield estimate (simple)
st.markdown("---")
st.header("Outputs & Results")
colA, colB, colC = st.columns(3)
colA.metric("Cell Efficiency (%)", f"{cell_eff:.3f}")
colB.metric("Module Efficiency (%)", f"{module_eff:.3f}")
colC.metric("Total CTM Loss (ppt)", f"{total_ctm_loss:.3f}")

# Breakdown dataframe
df_breakdown = pd.DataFrame({
    "Loss Type": ["Optical (shadowing+encap+refl)", "Resistive (contacts + fingers + busbars)", "Mismatch (cell variability)"],
    "Loss (%)": [optical_loss, resistive_loss, mismatch_loss]
})

st.subheader("CTM Loss Breakdown")
st.table(df_breakdown.style.format({"Loss (%)":"{:.3f}"}))

# Pie chart
fig1, ax1 = plt.subplots()
loss_vals = [optical_loss, resistive_loss, mismatch_loss]
labels = ["Optical", "Resistive", "Mismatch"]
ax1.pie(loss_vals, labels=labels, autopct='%1.1f%%', startangle=140)
ax1.axis('equal')
st.pyplot(fig1)

# Energy estimate
st.subheader("Energy Yield Impact (Estimate)")
area_m2 = ((wafer_x/1000.0) * (wafer_y/1000.0)) * num_cells  # module area (approx) assuming cells tiled tightly
baseline_irr_kwh = COEFF["baseline_irradiance_kwh_per_m2_per_year"]
module_power_kw_per_m2 = (module_eff / 100.0) * 1000.0  # W per m2 at STC approximated to kW/m2
annual_yield_kwh = (module_power_kw_per_m2 * area_m2) * (baseline_irr_kwh / 1000.0)  # rough yearly kWh
st.write(f"Approx. Module Area (m²): {area_m2:.3f}")
st.write(f"Rough Estimated Annual Energy (kWh/year): {annual_yield_kwh:.1f}")
st.caption("Note: This is a location-independent estimate. Replace baseline irradiance with site-specific data for accurate yield.")

# Download CSV summary
summary = {
    "cell_eff_%": cell_eff,
    "num_cells": num_cells,
    "wafer_x_mm": wafer_x,
    "wafer_y_mm": wafer_y,
    "optical_loss_%": optical_loss,
    "resistive_loss_%": resistive_loss,
    "mismatch_loss_%": mismatch_loss,
    "module_eff_%": module_eff,
    "annual_kwh_est": annual_yield_kwh
}
df_summary = pd.DataFrame([summary])

def get_table_download_link(df, filename="ctm_summary.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

st.download_button("Download results (CSV)", data=get_table_download_link(df_summary), file_name="ctm_summary.csv", mime="text/csv")

st.info("""
Model notes:
• This calculator uses approximations and empirical scaling factors. Use lab data to calibrate coefficients.
• Key coefficients to tune: shadow_factor, encap_penalty per encapsulant, contact_to_pct, finger_scale, busbar_factor_per_busbar.
• For production-grade CTM estimation, add spectral response (.xml) handling, detailed geometry meshes, and electrical circuit simulation (distributed series resistance + parallel leakage).
""")
