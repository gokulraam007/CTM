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
    page_title="CTM Loss Calculator - EON TOPCon",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# Cell-to-Module (CTM) Loss Calculator")
st.markdown("### EON TOPCon Technology | Luminous Power Technologies")

st.sidebar.header("Configuration")

if st.sidebar.button("Reset to EON TOPCon Defaults"):
    st.session_state.reset = True

st.sidebar.subheader("1. Solar Cell Parameters")
cell_power = st.sidebar.number_input("Cell Power (Wp)", min_value=4.0, max_value=7.0, value=5.5, step=0.1, help="EON TOPCon typical: 5.5 Wp")
cell_efficiency = st.sidebar.number_input("Cell Efficiency (%)", min_value=20.0, max_value=26.0, value=24.5, step=0.1, help="EON TOPCon: 24.5% (typical)")
num_cells = st.sidebar.number_input("Number of Cells", min_value=60, max_value=144, value=108, step=1, help="EON TOPCon: 108 cells")

st.sidebar.subheader("2. Module Geometry")
module_area = st.sidebar.number_input("Module Area (m²)", min_value=1.0, max_value=3.0, value=2.38, step=0.01, help="EON TOPCon: 2.38 m²")
cell_area = st.sidebar.number_input("Cell Area (cm²)", min_value=200.0, max_value=300.0, value=274.5, step=0.5, help="M10 cells: 274.5 cm²")

total_cell_area = (num_cells * cell_area) / 10000
inactive_area_fraction = 1 - (total_cell_area / module_area)

st.sidebar.subheader("3. Optical Loss Parameters")
glass_transmission = st.sidebar.slider("Glass Transmission (%)", 88.0, 96.0, 91.5, 0.5, help="EON: 91.5% (standard)")
eva_transmission = st.sidebar.slider("EVA Transmission (%)", 94.0, 98.0, 96.5, 0.5, help="EON: 96.5% (UV-stable)")

st.sidebar.subheader("4. Resistive Loss Parameters")
num_busbars = st.sidebar.selectbox("Number of Busbars", [3, 5, 9, 12, 16], index=2, help="EON: 9 busbars (MBB)")
ribbon_width = st.sidebar.number_input("Ribbon Width (mm)", min_value=0.8, max_value=2.5, value=1.5, step=0.1)
ribbon_thickness = st.sidebar.number_input("Ribbon Thickness (mm)", min_value=0.15, max_value=0.35, value=0.25, step=0.05)

st.sidebar.subheader("5. Mismatch Parameters")
cell_binning_tolerance = st.sidebar.slider("Cell Binning Tolerance (±%)", 0.0, 5.0, 2.0, 0.5, help="EON: ±2.0% (tight sorting)")

st.sidebar.subheader("6. Additional Parameters")
junction_box_loss = st.sidebar.slider("Junction Box & Cable Loss (%)", 0.1, 2.0, 0.5, 0.1)

geometric_loss = inactive_area_fraction * 100

glass_reflection_loss = (1 - glass_transmission/100) * 100
eva_absorption_loss = (1 - eva_transmission/100) * 100
optical_coupling_gain = 2.5
ribbon_coverage = (ribbon_width * num_busbars) / (np.sqrt(cell_area * 100))
ribbon_shading_loss = ribbon_coverage * 0.7
net_optical_loss = glass_reflection_loss + eva_absorption_loss + ribbon_shading_loss - optical_coupling_gain

finger_length_factor = 156 / num_busbars
base_resistive_loss = 0.8
resistive_loss = base_resistive_loss * (5 / num_busbars) ** 1.5
ribbon_resistivity = 1.7e-8
ribbon_area = ribbon_width * ribbon_thickness / 1e6
ribbon_resistance_factor = (ribbon_resistivity * 0.156) / ribbon_area
ribbon_loss_contribution = 0.3 * (ribbon_resistance_factor / 0.0001)
total_resistive_loss = resistive_loss + ribbon_loss_contribution

mismatch_loss = 0.5 + (cell_binning_tolerance / 2.0) * 0.3

jb_cable_loss = junction_box_loss

total_ctm_loss = geometric_loss + net_optical_loss + total_resistive_loss + mismatch_loss + jb_cable_loss

total_cell_power = cell_power * num_cells
ctm_ratio = 1 - (total_ctm_loss / 100)
module_power = total_cell_power * ctm_ratio
module_efficiency = (module_power / (module_area * 1000)) * 100

st.markdown("---")
st.markdown("## Key Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cell Power", f"{total_cell_power:.0f} Wp", "")

with col2:
    st.metric("Module Power", f"{module_power:.1f} Wp", f"{module_power - total_cell_power:.1f} W")

with col3:
    st.metric("Module Efficiency", f"{module_efficiency:.2f}%", f"{module_efficiency - cell_efficiency:.2f}%")

with col4:
    st.metric("CTM Ratio", f"{ctm_ratio*100:.2f}%", f"-{total_ctm_loss:.2f}%")

st.markdown("---")

st.markdown("## Loss Breakdown Analysis")

loss_data = {
    "Loss Category": [
        "Geometric (Inactive Area)",
        "Optical - Glass Reflection",
        "Optical - EVA Absorption",
        "Optical - Ribbon Shading",
        "Optical Coupling Gain",
        "Resistive (Cell + Ribbon)",
        "Mismatch (Binning)",
        "Junction Box & Cables",
        "TOTAL CTM LOSS"
    ],
    "Loss Value (%)": [
        f"{geometric_loss:.2f}",
        f"{glass_reflection_loss:.2f}",
        f"{eva_absorption_loss:.2f}",
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
        f"{-total_cell_power * eva_absorption_loss/100:.2f}",
        f"{-total_cell_power * ribbon_shading_loss/100:.2f}",
        f"+{total_cell_power * optical_coupling_gain/100:.2f}",
        f"{-total_cell_power * total_resistive_loss/100:.2f}",
        f"{-total_cell_power * mismatch_loss/100:.2f}",
        f"{-total_cell_power * jb_cable_loss/100:.2f}",
        f"{-(total_cell_power - module_power):.2f}"
    ]
}

df_losses = pd.DataFrame(loss_data)
st.dataframe(df_losses, use_container_width=True, hide_index=True)

st.markdown("---")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.markdown("### CTM Loss Waterfall Chart")

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Cell Power", "Geometric", "Glass", "EVA", "Ribbon Shade", "Coupling Gain", "Resistive", "Mismatch", "JB & Cable", "Module Power"]

    values = [
        total_cell_power,
        -total_cell_power * geometric_loss/100,
        -total_cell_power * glass_reflection_loss/100,
        -total_cell_power * eva_absorption_loss/100,
        -total_cell_power * ribbon_shading_loss/100,
        total_cell_power * optical_coupling_gain/100,
        -total_cell_power * total_resistive_loss/100,
        -total_cell_power * mismatch_loss/100,
        -total_cell_power * jb_cable_loss/100,
        module_power
    ]

    cumulative = [total_cell_power]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(module_power)

    colors_list = ["#2E7D32"] + ["#D32F2F" if v < 0 else "#1976D2" for v in values[1:-1]] + ["#2E7D32"]

    positions = []
    heights = []
    bottoms = []

    for i, (val, cum) in enumerate(zip(values, cumulative)):
        if i == 0 or i == len(values) - 1:
            heights.append(val)
            bottoms.append(0)
        else:
            heights.append(abs(val))
            bottoms.append(cum - (val if val > 0 else 0))
        positions.append(i)

    bars = ax.bar(positions, heights, bottom=bottoms, color=colors_list, edgecolor="black", linewidth=1.2, width=0.6)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold", rotation=45, ha="right")
    ax.set_ylabel("Power (Wp)", fontsize=11, fontweight="bold")
    ax.set_title("Power Waterfall: Cell to Module", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, total_cell_power * 1.1)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f"{height:.0f}W", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    plt.tight_layout()
    st.pyplot(fig)

with col_viz2:
    st.markdown("### Loss Distribution (Pie Chart)")

    pie_labels = ["Geometric", "Glass", "EVA", "Ribbon Shade", "Resistive", "Mismatch", "JB & Cable"]
    pie_values = [
        geometric_loss,
        glass_reflection_loss,
        eva_absorption_loss,
        ribbon_shading_loss,
        total_resistive_loss,
        mismatch_loss,
        jb_cable_loss
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, autopct="%1.1f%%", colors=["#FF6B6B", "#FFA500", "#FFD700", "#4ECDC4", "#FF69B4", "#87CEEB", "#98D8C8"], startangle=90, textprops={"fontsize": 10, "weight": "bold"})

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)
        autotext.set_weight("bold")

    ax2.set_title("Total CTM Loss Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")

def create_pdf_report(total_cell_power, module_power, module_efficiency, df_losses, geometric_loss, glass_reflection_loss, eva_absorption_loss, ribbon_shading_loss, optical_coupling_gain, total_resistive_loss, mismatch_loss, jb_cable_loss, total_ctm_loss, ctm_ratio):

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

    story = []

    story.append(Paragraph("CELL-TO-MODULE (CTM) LOSS ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 0.1*inch))

    company_text = "Luminous Power Technologies | EON TOPCon Technology"
    story.append(Paragraph(company_text, body_style))
    report_date = f"Report Generated: {datetime.now().strftime('%d %B %Y | %H:%M:%S IST')}"
    story.append(Paragraph(report_date, body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    summary_text = f"This report presents a detailed Cell-to-Module (CTM) loss analysis for EON TOPCon solar modules. The analysis identifies and quantifies all loss mechanisms between the individual solar cell efficiency and the final module efficiency. Total CTM loss: <b>{total_ctm_loss:.2f}%</b>, resulting in module power of <b>{module_power:.1f} Wp</b> from cell power of <b>{total_cell_power:.0f} Wp</b>."
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("KEY RESULTS", heading_style))
    results_data = [
        ["Parameter", "Value", "Unit"],
        ["Cell Power", f"{total_cell_power:.0f}", "Wp"],
        ["Module Power", f"{module_power:.1f}", "Wp"],
        ["Power Loss", f"{total_cell_power - module_power:.1f}", "Wp"],
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

    story.append(Paragraph("DETAILED LOSS BREAKDOWN", heading_style))

    loss_breakdown_data = [
        ["Loss Category", "Loss (%)", "Power Impact (W)"],
        ["Geometric (Inactive Area)", f"{geometric_loss:.2f}", f"{-total_cell_power * geometric_loss/100:.2f}"],
        ["Optical - Glass Reflection", f"{glass_reflection_loss:.2f}", f"{-total_cell_power * glass_reflection_loss/100:.2f}"],
        ["Optical - EVA Absorption", f"{eva_absorption_loss:.2f}", f"{-total_cell_power * eva_absorption_loss/100:.2f}"],
        ["Optical - Ribbon Shading", f"{ribbon_shading_loss:.2f}", f"{-total_cell_power * ribbon_shading_loss/100:.2f}"],
        ["Optical Coupling Gain", f"-{optical_coupling_gain:.2f}", f"+{total_cell_power * optical_coupling_gain/100:.2f}"],
        ["Resistive (Cell + Ribbon)", f"{total_resistive_loss:.2f}", f"{-total_cell_power * total_resistive_loss/100:.2f}"],
        ["Mismatch (Binning)", f"{mismatch_loss:.2f}", f"{-total_cell_power * mismatch_loss/100:.2f}"],
        ["Junction Box & Cables", f"{jb_cable_loss:.2f}", f"{-total_cell_power * jb_cable_loss/100:.2f}"],
        ["TOTAL", f"{total_ctm_loss:.2f}", f"{-total_cell_power + module_power:.2f}"]
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
    story.append(Spacer(1, 0.2*inch))

    story.append(PageBreak())
    story.append(Paragraph("SCIENTIFIC REFERENCES", heading_style))

    references = "1. Haedrich, I., et al. (2014). Unified methodology for determining CTM ratios. Solar Energy Materials and Solar Cells, 131, 14-23.<br/><br/>2. Mittag, M., et al. (2019). Cell-to-Module (CTM) Analysis for Photovoltaic Modules. IEEE PVSC Conference.<br/><br/>3. Hanifi, H., et al. (2017). Investigation of cell-to-module (CTM) ratios of PV modules by analysis of loss and gain mechanisms. Photovoltaics International.<br/><br/>4. Fraunhofer ISE (2020). SmartCalc.CTM - Cell to Module Analysis Software. www.cell-to-module.com<br/><br/>5. IEC 61215 (2021). Terrestrial photovoltaic (PV) modules - Design qualification and type approval.<br/><br/>6. Roy, J.N., et al. (2016). Comprehensive analysis and modeling of cell to module (CTM) conversion loss. Solar Energy, 135, 618-628."

    story.append(Paragraph(references, body_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("_" * 80, body_style))
    story.append(Spacer(1, 0.1*inch))

    signature_text = f"<b>Gokul Raam G</b><br/>Senior Engineer, R&D<br/>Luminous Power Technologies<br/>(Schneider Electric)<br/><br/><i>Date: {datetime.now().strftime('%d %B %Y')}</i>"

    story.append(Paragraph(signature_text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

st.markdown("---")
st.markdown("## Download Report")

col_download1, col_download2 = st.columns([1, 1])

with col_download1:
    if st.button("Generate PDF Report", key="generate_pdf", use_container_width=True):
        pdf_buffer = create_pdf_report(total_cell_power, module_power, module_efficiency, df_losses, geometric_loss, glass_reflection_loss, eva_absorption_loss, ribbon_shading_loss, optical_coupling_gain, total_resistive_loss, mismatch_loss, jb_cable_loss, total_ctm_loss, ctm_ratio)

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=f"CTM_Analysis_Report_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pdf",
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
### Scientific References

1. **Haedrich, I., et al. (2014).** "Unified methodology for determining CTM ratios." *Solar Energy Materials and Solar Cells*, 131, 14-23.

2. **Mittag, M., et al. (2019).** "Cell-to-Module (CTM) Analysis for Photovoltaic Modules." *IEEE PVSC Conference*.

3. **Hanifi, H., et al. (2017).** "Investigation of cell-to-module (CTM) ratios of PV modules by analysis of loss and gain mechanisms." *Photovoltaics International*.

4. **Fraunhofer ISE (2020).** SmartCalc.CTM - Cell to Module Analysis Software. www.cell-to-module.com

5. **IEC 61215 (2021).** Terrestrial photovoltaic (PV) modules - Design qualification and type approval.

6. **Roy, J.N., et al. (2016).** "Comprehensive analysis and modeling of cell to module (CTM) conversion loss." *Solar Energy*, 135, 618-628.

---

**Developed by:** Gokul Raam G | Senior Engineer, R&D | Luminous Power Technologies (Schneider Electric)
""")
