import streamlit as st
from PIL import Image, ImageChops, ImageOps
import numpy as np
import io
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Satellite Semantic Change Screening | Team QUADX",
    page_icon="🛰️",
    layout="wide"
)

# Custom Clean Navy Blue & Black Modern Theme Styling (Dots Removed)
st.markdown("""
    <style>
    .stApp {
        background-color: #030712;
        color: #F8FAFC;
    }
    @keyframes floatSatellite {
        0% { transform: translateX(-60px) translateY(10px) rotate(-10deg); opacity: 0.7; }
        50% { transform: translateX(calc(100vw - 150px)) translateY(-15px) rotate(10deg); opacity: 1; }
        100% { transform: translateX(calc(100vw + 60px)) translateY(10px) rotate(-10deg); opacity: 0.7; }
    }
    .orbiting-satellite {
        position: absolute;
        top: 15px;
        left: 0px;
        font-size: 2.5rem;
        animation: floatSatellite 18s linear infinite;
        pointer-events: none;
        z-index: 2;
        filter: drop-shadow(0 0 8px rgba(96, 165, 250, 0.6));
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
    }
    .team-badge {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }
    .disclaimer-box {
        background-color: #172554;
        border: 1px solid #3B82F6;
        color: #93C5FD;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Header & Team Branding with Animated Satellite
st.markdown("""
    <div class="main-header">
        <div class="orbiting-satellite">🛰️</div>
        <span class="team-badge">Team QUADX — Semantic EO Intelligence</span>
        <h1 style="color: #60A5FA; margin-top: 12px;">Satellite Image Semantic Change Screening</h1>
        <p style="color: #94A3B8; font-size: 1.1rem;">Multi-Temporal Earth Observation Analysis with Automated Change Attribution & Reporting</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls & Parameters
st.sidebar.markdown("## 🎛️ Screening Controls")
st.sidebar.markdown("---")
sensitivity = st.sidebar.slider(
    "Change Detection Threshold", 
    min_value=10, 
    max_value=100, 
    value=40, 
    step=5,
    help="Adjusts sensitivity to pixel intensity variation between timelines."
)

overlay_mode = st.sidebar.selectbox(
    "Display Overlay Mode",
    ["Semantic Classified Mask", "Difference Mask", "Heatmap Overlay", "Side-by-Side Split"]
)

show_contours = st.sidebar.checkbox("Highlight High-Intensity Clusters", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Metadata & Source Info")
img1_date = st.sidebar.date_input("Timeline 1 Date", value=None)
img2_date = st.sidebar.date_input("Timeline 2 Date", value=None)
source_sat = st.sidebar.selectbox("Satellite Source", ["Sentinel-2 (ESA)", "Landsat 8/9 (NASA/USGS)", "PlanetScope (Commercial)", "Custom Raster"])

# Main Layout: Image Uploaders
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Timeline 1 (Baseline)")
    file1 = st.file_uploader("Upload Baseline Image", type=["png", "jpg", "jpeg", "tif"], key="t1")

with col2:
    st.markdown("### 📅 Timeline 2 (Target / Recent)")
    file2 = st.file_uploader("Upload Target Image", type=["png", "jpg", "jpeg", "tif"], key="t2")

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")
    
    # Ensure images match dimensions for comparison
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    # Convert to numpy arrays for processing
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Compute absolute difference
    diff = np.abs(arr1 - arr2)
    gray_diff = np.mean(diff, axis=2)
    
    # Candidate change mask based on threshold
    mask = gray_diff > sensitivity
    changed_pixels = np.sum(mask)
    total_pixels = mask.size
    change_percentage = (changed_pixels / total_pixels) * 100

    # ==========================================
    # SEMANTIC CHANGE CLASSIFICATION HEURISTICS
    # ==========================================
    r2, g2, b2 = arr2[..., 0], arr2[..., 1], arr2[..., 2]
    brightness2 = (r2 + g2 + b2) / 3.0
    
    is_water_target = (b2 > r2) & (g2 >= r2) & (brightness2 < 140)
    is_urban_target = (brightness2 > 160) & (np.abs(r2 - g2) < 25) & (np.abs(g2 - b2) < 25)
    is_veg_target = (g2 > r2) & (g2 > b2)

    changed_water = mask & is_water_target
    changed_urban = mask & is_urban_target
    changed_veg = mask & is_veg_target
    changed_other = mask & ~(changed_water | changed_urban | changed_veg)

    water_count = np.sum(changed_water)
    urban_count = np.sum(changed_urban)
    veg_count = np.sum(changed_veg)
    other_count = np.sum(changed_other)

    safe_denom = changed_pixels if changed_pixels > 0 else 1
    water_pct = (water_count / safe_denom) * 100
    urban_pct = (urban_count / safe_denom) * 100
    veg_pct = (veg_count / safe_denom) * 100
    other_pct = (other_count / safe_denom) * 100

    # Build Semantic Classified RGB Mask for Visualization
    semantic_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    semantic_rgb[changed_water] = [30, 144, 255]   # Dodger Blue (Water)
    semantic_rgb[changed_urban] = [255, 69, 0]     # Orange Red (Urban / Buildings)
    semantic_rgb[changed_veg]   = [46, 139, 87]    # Sea Green (Vegetation shift)
    semantic_rgb[changed_other] = [218, 165, 32]   # Goldenrod (Other / Bare soil)
    
    semantic_img = Image.fromarray(semantic_rgb)
    standard_mask_rgb = np.stack([mask*255, mask*50, mask*50], axis=-1).astype(np.uint8)
    standard_mask_img = Image.fromarray(standard_mask_rgb)

    st.markdown("---")
    st.markdown("## 📊 Quantitative & Semantic Screening Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Altered Area", value=f"{change_percentage:.2f}%", delta=f"{changed_pixels:,} px")
    with m2:
        st.metric(label="🏗️ New Built-Up / Urban", value=f"{urban_pct:.1f}%", delta=f"{urban_count:,} px")
    with m3:
        st.metric(label="💧 Water Body Shift", value=f"{water_pct:.1f}%", delta=f"{water_count:,} px")
    with m4:
        st.metric(label="🌲 Vegetation Alteration", value=f"{veg_pct:.1f}%", delta=f"{veg_count:,} px")

    st.markdown("---")
    st.markdown("## 🔍 Visual Analysis & Semantic Attribution")

    if overlay_mode == "Semantic Classified Mask":
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img2, caption=f"Target Image ({img2_date or 'T2'})", width="stretch")
        with col_b:
            st.image(semantic_img, caption="Semantic Attribution Map (Blue:Water | Orange:Urban | Green:Veg | Gold:Other)", width="stretch")

    elif overlay_mode == "Difference Mask":
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img1, caption=f"Baseline Image ({img1_date or 'T1'})", width="stretch")
        with col_b:
            st.image(standard_mask_img, caption="Raw Candidate Change Mask", width="stretch")

    elif overlay_mode == "Heatmap Overlay":
        blended = Image.blend(img2, semantic_img, alpha=0.5)
        st.image(blended, caption="Semantic Attribution Overlay on Target Image", width="stretch")

    else: # Side-by-Side
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img1, caption=f"Baseline ({img1_date or 'T1'})", width="stretch")
        with col_b:
            st.image(img2, caption=f"Target ({img2_date or 'T2'})", width="stretch")
        st.image(semantic_img, caption="Classified Change Regions", width="stretch")

    # Export Section (Image & CSV Report)
    st.markdown("---")
    st.markdown("### 💾 Export Results & Reporting")
    
    buf = io.BytesIO()
    semantic_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # Compile report dictionary for CSV export
    report_data = {
        "Parameter / Metric": [
            "Baseline Date (T1)", "Target Date (T2)", "Satellite Source", 
            "Sensitivity Threshold", "Image Resolution (px)", "Total Altered Area (%)", 
            "Detected Anomalous Pixels", "New Built-Up / Urban (%)", 
            "Water Body Shift (%)", "Vegetation Alteration (%)", "Other / Unclassified (%)"
        ],
        "Value": [
            str(img1_date or "N/A"), str(img2_date or "N/A"), source_sat,
            sensitivity, f"{img1.size[0]} × {img1.size[1]}", f"{change_percentage:.2f}%",
            f"{changed_pixels:,}", f"{urban_pct:.2f}% ({urban_count:,} pixels)",
            f"{water_pct:.2f}% ({water_count:,} pixels)", f"{veg_pct:.2f}% ({veg_count:,} pixels)",
            f"{other_pct:.2f}% ({other_count:,} pixels)"
        ]
    }
    df_report = pd.DataFrame(report_data)
    csv_bytes = df_report.to_csv(index=False).encode('utf-8')

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            label="📥 Download Semantic Attribution Map (PNG)",
            data=byte_im,
            file_name="quadx_semantic_change_map.png",
            mime="image/png"
        )
    with exp_col2:
        st.download_button(
            label="📊 Download Metrics Report (CSV)",
            data=csv_bytes,
            file_name="quadx_change_screening_report.csv",
            mime="text/csv"
        )

    with st.expander("🔍 Preview CSV Report Data"):
        st.dataframe(df_report, width="stretch")

    # Documentation & Confounders Panel
    st.markdown("---")
    st.markdown("### 📑 Preprocessing & Attribution Confidence Notes")
    
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        st.markdown("""
        *Semantic Heuristic Rules:*
        * *Urban/Buildings:* Identified via high reflectance and neutral RGB balance (concrete/roof signatures).
        * *Water Bodies:* Isolated using blue-green chromatic dominance and lower brightness levels.
        * *Vegetation:* Extracted via green channel radiometric superiority.
        """)
    with doc_col2:
        st.markdown("""
        *Likely Confounders & False Attribution Risks:*
        * High sun-angle glare on metal roofs mimicking water signatures.
        * Seasonal agricultural harvesting looking like bare soil/urban expansion.
        * Atmospheric haze altering absolute RGB intensity thresholds.
        """)

    # Mandatory Safety Disclaimer
    st.markdown("""
        <div class="disclaimer-box">
            <b>⚠️ MANDATORY VALIDATION NOTICE:</b> Semantic classes above are heuristic approximations derived from optical spectral profiles. 
            Do not treat automated attribution as definitive ground truth without secondary GIS verification or field inspection.
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Please upload both Baseline (Timeline 1) and Target (Timeline 2) satellite images in the sidebar / panels above to initiate semantic screening.")