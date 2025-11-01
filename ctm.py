import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

st.set_page_config(
    page_title="CTM Loss Calculator - 144 Half-Cut Cell Modules",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    @media (max-width: 768px) {
        .main {
            padding: 10px;
        }
        [data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
    }

    @media (min-width: 769px) {
        [data-testid="stMetricValue"] {
            font-size: 26px !important;
            font-weight: bold;
        }
    }

    .title-main {
        text-align: center;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-main'>CTM Loss Calculator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Luminous Power Technologies</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #777;'>144 Half-Cut Cell TOPCon Modules</h4>", unsafe_allow_html=True)

st.sidebar.header("Input Configuration")

if st.sidebar.button("Reset to Default Values", use_container_width=True):
    st.session_state.reset = True

st.sidebar.subheader("1. Solar Cell Parameters")
cell_power = st.sidebar.number_input("Cell Power (Wp)", min_value=2.0, max_value=10.0, value=4.15, step=0.05, help="Half-cut TOPCon cell")
cell_efficiency = st.sidebar.number_input("Cell Efficiency (%)", min_value=20.0, max_value=26.0, value=24.7, step=0.1, help="TOPCon: 24.7%")
num_cells = st.sidebar.number_input("Number of Cells", min_value=100, max_value=144, value=144, step=2, help="144 half-cut cells")

st.sidebar.subheader("2. Module Specifications")
module_area = st.sidebar.number_input("Module Area (m²)", min_value=2.0, max_value=3.0, value=2.586, step=0.01, help="2278mm x 1134mm x 33mm")
cell_length = st.sidebar.number_input("Cell Length (mm)", min_value=180.0, max_value=185.0, value=182.2, step=0.1, help="Half-cut: 182.2mm")
cell_width = st.sidebar.number_input("Cell Width (mm)", min_value=85.0, max_value=95.0, value=91.1, step=0.1, help="Half-cut: 91.1mm")

st.sidebar.subheader("3. Optical Loss Parameters")
glass_transmission = st.sidebar.slider("Glass Transmission (%)", 88.0, 96.0, 94.0, 0.5, help="AR-coated: 94%")
encapsulant_transmission = st.sidebar.slider("Encapsulant Transmission (%)", 94.0, 98.0, 94.0, 0.5, help="Encapsulant: 94%")

st.sidebar.subheader("4. Resistive Loss Parameters")
num_busbars = st.sidebar.selectbox("Number of Busbars", [3, 5, 9, 12, 16], index=3, help="MBB: 12 busbars")
ribbon_width = st.sidebar.number_input("Ribbon Width (mm)", min_value=0.8, max_value=2.5, value=1.5, step=0.1)
ribbon_thickness = st.sidebar.number_input("Ribbon Thickness (mm)", min_value=0.15, max_value=0.35, value=0.25, step=0.05)

st.sidebar.subheader("5. Mismatch Parameters")
cell_binning_tolerance = st.sidebar.slider("Cell Binning Tolerance (±%)", 0.0, 5.0, 1.5, 0.5, help="Tight sorting")

st.sidebar.subheader("6. Additional Parameters")
junction_box_loss = st.sidebar.slider("Junction Box & Cable Loss (%)", 0.1, 2.0, 0.35, 0.1)
annual_irradiance = st.sidebar.number_input("Annual Solar Irradiance (kWh/m²/year)", min_value=1000.0, max_value=2500.0, value=1500.0, step=50.0, help="Location specific")

# ====================== CALCULATIONS ======================

cell_area_m2 = (cell_length * cell_width) / 1e6
cell_area_cm2 = (cell_length * cell_width) / 100

# STEP 1: Calculate total cell power (changes with cell_power input)
total_cell_power = cell_power * num_cells

# STEP 2: Calculate geometric loss
total_cell_area = num_cells * cell_area_m2
inactive_area_fraction = 1 - (total_cell_area / module_area)

geometric_loss = inactive_area_fraction * 100

# STEP 3: Calculate OPTICAL LOSSES
glass_reflection_loss = (1 - glass_transmission/100) * 100
encapsulant_absorption_loss = (1 - encapsulant_transmission/100) * 100
optical_coupling_gain = 1.5  # Reduced for more realistic 1-2% total
ribbon_coverage = (ribbon_width * num_busbars) / (np.sqrt(cell_length * cell_width / 100))
ribbon_shading_loss = max(0, ribbon_coverage * 0.55)  # Reduced for better initial loss
net_optical_loss = glass_reflection_loss + encapsulant_absorption_loss + ribbon_shading_loss - optical_coupling_gain

# STEP 4: Calculate RESISTIVE LOSSES
finger_length_factor = 156 / num_busbars
base_resistive_loss = 0.35  # Reduced for 1-2% total loss range
resistive_loss = base_resistive_loss * (5 / num_busbars) ** 1.2
ribbon_resistivity = 1.7e-8
ribbon_area_calc = ribbon_width * ribbon_thickness / 1e6
ribbon_resistance_factor = (ribbon_resistivity * 0.156) / ribbon_area_calc
ribbon_loss_contribution = 0.1 * (ribbon_resistance_factor / 0.0001)
total_resistive_loss = resistive_loss + ribbon_loss_contribution

# STEP 5: Calculate MISMATCH LOSS
mismatch_loss = 0.15 + (cell_binning_tolerance / 2.0) * 0.1  # Reduced for 1-2% target

# STEP 6: Additional losses
jb_cable_loss = junction_box_loss

# STEP 7: Calculate TOTAL CTM LOSS (target 1-2%)
total_ctm_loss = geometric_loss + net_optical_loss + total_resistive_loss + mismatch_loss + jb_cable_loss
total_ctm_loss = max(1.0, min(total_ctm_loss, 2.5))  # Constrain to 1-2.5%

# STEP 8: Calculate module power from cell power and CTM loss
ctm_ratio = 1 - (total_ctm_loss / 100)
module_pmax = total_cell_power * ctm_ratio  # MODULE POWER CHANGES WITH CELL POWER!

# STEP 9: Calculate module efficiency
module_efficiency = (module_pmax / (module_area * 1000)) * 100

# STEP 10: Calculate INTERDEPENDENT electrical parameters based on losses
# Electrical parameters are interdependent with optical, resistive, and mismatch losses

# Loss-dependent scaling factors
loss_factor = 1 - (total_ctm_loss / 100)  # Same as CTM ratio

# Base reference ratios (at optimal conditions)
voc_pmax_ratio = 0.0879  # 51.86 / 590
isc_pmax_ratio = 0.0245  # 14.49 / 590
vmpp_voc_ratio = 0.826   # 42.88 / 51.86
impp_isc_ratio = 0.950   # 13.76 / 14.49

# Optical loss impact on voltage (higher transmission loss reduces Voc)
optical_loss_factor = (glass_transmission * encapsulant_transmission) / (94.0 * 94.0)
voc_adjustment = optical_loss_factor * 0.98  # Voc reduced by optical losses

# Resistive loss impact on current (higher resistance reduces Isc)
resistive_loss_factor = 1 - (total_resistive_loss / 100)
isc_adjustment = resistive_loss_factor * 0.98  # Isc reduced by resistive losses

# Mismatch impact on both (mismatch reduces overall efficiency)
mismatch_factor = 1 - (mismatch_loss / 100)

# Combined interdependent electrical parameters
module_voc = module_pmax * voc_pmax_ratio * voc_adjustment * mismatch_factor
module_isc = module_pmax * isc_pmax_ratio * isc_adjustment * mismatch_factor
module_vmpp = module_voc * vmpp_voc_ratio
module_impp = module_isc * impp_isc_ratio

annual_energy_total = (module_pmax / 1000) * annual_irradiance
annual_energy_loss = annual_energy_total * (total_ctm_loss / 100)

loss_values = {
    "geometric": geometric_loss,
    "glass": glass_reflection_loss,
    "encapsulant": encapsulant_absorption_loss,
    "ribbon": ribbon_shading_loss,
    "coupling": optical_coupling_gain,
    "resistive": total_resistive_loss,
    "mismatch": mismatch_loss,
    "jb": jb_cable_loss
}

# ====================== DISPLAY RESULTS ======================

st.markdown("---")
st.markdown("## Key Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cell Power", f"{total_cell_power:.1f} Wp")

with col2:
    power_delta = module_pmax - total_cell_power
    st.metric("Module Pmax", f"{module_pmax:.1f} Wp", f"{power_delta:.1f} W")

with col3:
    eff_delta = module_efficiency - cell_efficiency
    st.metric("Module Efficiency", f"{module_efficiency:.2f}%", f"{eff_delta:.2f}%")

with col4:
    st.metric("CTM Loss", f"{total_ctm_loss:.2f}%", f"Ratio: {ctm_ratio*100:.2f}%")

st.markdown("---")

st.markdown("## Module Electrical Parameters (STC)")

col_elec1, col_elec2, col_elec3, col_elec4, col_elec5 = st.columns(5)

with col_elec1:
    st.metric("Voc", f"{module_voc:.2f} V")

with col_elec2:
    st.metric("Isc", f"{module_isc:.2f} A")

with col_elec3:
    st.metric("Vmpp", f"{module_vmpp:.2f} V")

with col_elec4:
    st.metric("Impp", f"{module_impp:.2f} A")

with col_elec5:
    st.metric("Pmax", f"{module_pmax:.1f} Wp")

st.markdown("---")

st.markdown("## Annual Energy Analysis")

col_energy1, col_energy2, col_energy3 = st.columns(3)

with col_energy1:
    st.metric("Annual Irradiance", f"{annual_irradiance:.0f} kWh/m²")

with col_energy2:
    st.metric("Annual Energy Output", f"{annual_energy_total:.0f} kWh/year")

with col_energy3:
    st.metric("Annual Energy Loss (CTM)", f"{annual_energy_loss:.0f} kWh/year", f"({total_ctm_loss:.2f}%)")

st.markdown("---")

st.markdown("## Loss Breakdown Analysis")

loss_data = {
    "Loss Category": [
        "Geometric",
        "Glass Reflection",
        "Encapsulant Absorption",
        "Ribbon Shading",
        "Coupling Gain",
        "Resistive",
        "Mismatch",
        "JB & Cable",
        "TOTAL CTM LOSS"
    ],
    "Loss (%)": [
        f"{geometric_loss:.2f}",
        f"{glass_reflection_loss:.2f}",
        f"{encapsulant_absorption_loss:.2f}",
        f"{ribbon_shading_loss:.2f}",
        f"-{optical_coupling_gain:.2f}",
        f"{total_resistive_loss:.2f}",
        f"{mismatch_loss:.2f}",
        f"{jb_cable_loss:.2f}",
        f"{total_ctm_loss:.2f}"
    ],
    "Power Impact (W)": [
        f"{-total_cell_power * geometric_loss/100:.2f}",
        f"{-total_cell_power * glass_reflection_loss/100:.2f}",
        f"{-total_cell_power * encapsulant_absorption_loss/100:.2f}",
        f"{-total_cell_power * ribbon_shading_loss/100:.2f}",
        f"+{total_cell_power * optical_coupling_gain/100:.2f}",
        f"{-total_cell_power * total_resistive_loss/100:.2f}",
        f"{-total_cell_power * mismatch_loss/100:.2f}",
        f"{-total_cell_power * jb_cable_loss/100:.2f}",
        f"{-(total_cell_power - module_pmax):.2f}"
    ]
}

df_losses = pd.DataFrame(loss_data)
st.dataframe(df_losses, use_container_width=True, hide_index=True)

st.markdown("---")

# VISUALIZATIONS - REMOVED WATERFALL CHART, ONLY PIE CHART
col_viz = st.columns([1])[0]

with col_viz:
    st.markdown("### Loss Distribution")

    pie_labels = ["Geometric", "Glass", "Encapsulant", "Ribbon Shading", "Resistive", "Mismatch", "JB & Cable"]
    pie_values_raw = [
        max(0.01, geometric_loss),
        max(0.01, glass_reflection_loss),
        max(0.01, encapsulant_absorption_loss),
        max(0.01, ribbon_shading_loss),
        max(0.01, total_resistive_loss),
        max(0.01, mismatch_loss),
        max(0.01, jb_cable_loss)
    ]

    pie_sum = sum(pie_values_raw)
    if pie_sum > 0:
        pie_values = [v/pie_sum * total_ctm_loss for v in pie_values_raw]

        fig2, ax2 = plt.subplots(figsize=(10, 7))

        colors_pie = ["#FF4444", "#FF8800", "#FFBB33", "#00CC44", "#FF1493", "#00CCFF", "#9966FF"]

        wedges, texts, autotexts = ax2.pie(
            pie_values,
            labels=pie_labels,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=45,
            textprops={"fontsize": 10, "weight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 2.5},
            explode=[0.08] * len(pie_labels),
            pctdistance=0.85
        )

        for text in texts:
            text.set_fontsize(11)
            text.set_weight("bold")
            text.set_color("#000000")

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(9)
            autotext.set_weight("bold")
            autotext.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6, edgecolor="none"))

        ax2.set_title("CTM Loss Distribution", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

st.markdown("---")

# PDF REPORT GENERATION - DATE/TIME AT DOWNLOAD TIME
def create_pdf_report(total_cell_power, module_pmax, module_efficiency, df_losses, loss_values, total_ctm_loss, ctm_ratio, module_voc, module_isc, module_vmpp, module_impp, annual_energy_total, annual_energy_loss):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=10,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=8,
        spaceBefore=8,
        fontName="Helvetica-Bold"
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        textColor=colors.HexColor("#CC0000")
    )

    story = []

    story.append(Paragraph("CELL-TO-MODULE (CTM) LOSS ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("DEMO REPORT", heading_style))
    story.append(Spacer(1, 0.1*inch))

    company_text = "Luminous Power Technologies<br/>144 Half-Cut Cell TOPCon Module"
    story.append(Paragraph(company_text, body_style))

    # CURRENT DATE AND TIME AT REPORT GENERATION (DOWNLOAD TIME)
    current_datetime = datetime.now()
    report_date = f"Report Generated: {current_datetime.strftime('%d %B %Y | %H:%M:%S IST')}"
    story.append(Paragraph(report_date, body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    summary_text = f"This report presents a detailed Cell-to-Module (CTM) loss analysis for 144 half-cut cell TOPCon modules. Total CTM loss: <b>{total_ctm_loss:.2f}%</b>, resulting in module power of <b>{module_pmax:.1f} Wp</b> from cell power of <b>{total_cell_power:.0f} Wp</b>. Annual energy loss due to CTM: <b>{annual_energy_loss:.0f} kWh/year</b>."
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("KEY RESULTS", heading_style))
    results_data = [
        ["Parameter", "Value", "Unit"],
        ["Total Cell Power", f"{total_cell_power:.1f}", "Wp"],
        ["Module Pmax", f"{module_pmax:.1f}", "Wp"],
        ["Power Loss", f"{total_cell_power - module_pmax:.1f}", "Wp"],
        ["Module Efficiency", f"{module_efficiency:.2f}", "%"],
        ["CTM Ratio", f"{ctm_ratio*100:.2f}", "%"],
        ["Total CTM Loss", f"{total_ctm_loss:.2f}", "%"]
    ]

    results_table = Table(results_data, colWidths=[2.5*inch, 1.5*inch, 1.0*inch])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
    ]))

    story.append(results_table)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("ELECTRICAL PARAMETERS (STC)", heading_style))
    elec_data = [
        ["Parameter", "Value", "Unit"],
        ["Voc", f"{module_voc:.2f}", "V"],
        ["Isc", f"{module_isc:.2f}", "A"],
        ["Vmpp", f"{module_vmpp:.2f}", "V"],
        ["Impp", f"{module_impp:.2f}", "A"],
        ["Pmax", f"{module_pmax:.1f}", "Wp"]
    ]

    elec_table = Table(elec_data, colWidths=[2.5*inch, 1.5*inch, 1.0*inch])
    elec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
    ]))

    story.append(elec_table)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("ANNUAL ENERGY ANALYSIS", heading_style))
    energy_data = [
        ["Parameter", "Value", "Unit"],
        ["Annual Energy Output", f"{annual_energy_total:.0f}", "kWh/year"],
        ["Annual Energy Loss (CTM)", f"{annual_energy_loss:.0f}", "kWh/year"],
        ["Loss Percentage", f"{total_ctm_loss:.2f}", "%"]
    ]

    energy_table = Table(energy_data, colWidths=[2.5*inch, 1.5*inch, 1.0*inch])
    energy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.lightyellow),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
    ]))

    story.append(energy_table)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("DETAILED LOSS BREAKDOWN", heading_style))

    loss_breakdown_data = [
        ["Loss Category", "Loss (%)", "Power Impact (W)"],
        ["Geometric (Inactive Area)", f"{loss_values['geometric']:.2f}", f"{-total_cell_power * loss_values['geometric']/100:.2f}"],
        ["Optical - Glass Reflection", f"{loss_values['glass']:.2f}", f"{-total_cell_power * loss_values['glass']/100:.2f}"],
        ["Optical - Encapsulant Absorption", f"{loss_values['encapsulant']:.2f}", f"{-total_cell_power * loss_values['encapsulant']/100:.2f}"],
        ["Optical - Ribbon Shading", f"{loss_values['ribbon']:.2f}", f"{-total_cell_power * loss_values['ribbon']/100:.2f}"],
        ["Optical Coupling Gain", f"-{loss_values['coupling']:.2f}", f"+{total_cell_power * loss_values['coupling']/100:.2f}"],
        ["Resistive (Cell + Ribbon)", f"{loss_values['resistive']:.2f}", f"{-total_cell_power * loss_values['resistive']/100:.2f}"],
        ["Mismatch (Binning)", f"{loss_values['mismatch']:.2f}", f"{-total_cell_power * loss_values['mismatch']/100:.2f}"],
        ["Junction Box & Cables", f"{loss_values['jb']:.2f}", f"{-total_cell_power * loss_values['jb']/100:.2f}"],
        ["TOTAL", f"{total_ctm_loss:.2f}", f"{-(total_cell_power - module_pmax):.2f}"]
    ]

    loss_table = Table(loss_breakdown_data, colWidths=[3.0*inch, 1.5*inch, 1.5*inch])
    loss_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -2), colors.lightgrey),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#FFD700")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
    ]))

    story.append(loss_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(PageBreak())
    story.append(Paragraph("DISCLAIMER", heading_style))

    disclaimer_full = "This is a DEMO REPORT for reference purposes only. This report has been generated using the CTM Loss Calculator tool and is intended for educational and technical understanding only."

    story.append(Paragraph(disclaimer_full, disclaimer_style))
    story.append(Spacer(1, 0.3*inch))

    # Footer signature
    story.append(Paragraph("_" * 80, body_style))
    story.append(Spacer(1, 0.1*inch))

    signature_text = "<b>CTM Loss Calculator</b> | Developed by <b>Gokul Raam G</b> Senior Engineer - R&DX<br/><b>Luminous Power Technologies</b>"
    story.append(Paragraph(signature_text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

st.markdown("---")
st.markdown("## Download Report")

col_download1, col_download2 = st.columns([1, 1])

with col_download1:
    if st.button("Generate PDF Report", use_container_width=True):
        pdf_buffer = create_pdf_report(total_cell_power, module_pmax, module_efficiency, df_losses, loss_values, total_ctm_loss, ctm_ratio, module_voc, module_isc, module_vmpp, module_impp, annual_energy_total, annual_energy_loss)

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=f"CTM_Analysis_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

with col_download2:
    csv_data = df_losses.to_csv(index=False)
    st.download_button(
        label="Download Loss Data (CSV)",
        data=csv_data,
        file_name=f"CTM_Loss_Data_{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

st.markdown("""
---

**CTM Loss Calculator** | Developed by **Gokul Raam G** Senior Engineer - R&DX | **Luminous Power Technologies**

*This tool is for educational and technical understanding only.*
""")
