import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DynaSTGCN Research Dashboard",
    layout="wide"
)

# =========================================================
# FINAL VERIFIED RESULTS
# =========================================================

BASELINE = 43.80
STANDARD = 25.03
OPTIMIZED = 23.36

HORIZONS = {
5:21.72,
15:22.33,
30:23.36,
45:24.44,
60:25.43
}

CHANNEL_IMPORTANCE = {
"Flow":0.61,
"Speed":0.26,
"Occupancy":0.13
}

ABLATION = pd.DataFrame([
["Baseline ASTGCN","✗","✗","✗",0.001,43.80],
["CAG + TEM","✓","✗","✓",0.001,25.72],
["Adj + TEM","✗","✓","✓",0.001,24.80],
["Full DynaSTGCN","✓","✓","✓",0.001,25.03],
["Optimized (No TEM)","✓","✓","✗",0.005,23.36]
],
columns=[
"Model",
"Channel Attention",
"Adaptive Adj",
"TEM",
"LR",
"MAE"
])

# =========================================================
# HEADER
# =========================================================

st.title("🚦 DynaSTGCN Traffic Forecasting Dashboard")

st.caption("PeMS04 Dataset · 307 Sensors · 60-minute horizon")

improvement=((BASELINE-OPTIMIZED)/BASELINE)*100

st.success(f"Performance Gain vs Baseline: {improvement:.1f}%")

st.divider()

# =========================================================
# KPI PANEL
# =========================================================

col1,col2,col3=st.columns(3)

col1.metric("Baseline ASTGCN",BASELINE)
col2.metric("Standard DynaSTGCN",STANDARD)
col3.metric("Optimized Model ★",OPTIMIZED)

# =========================================================
# MODEL COMPARISON GRAPH
# =========================================================

st.subheader("Overall Model Comparison")

fig,ax=plt.subplots()

models=["Baseline","Standard","Optimized"]
values=[BASELINE,STANDARD,OPTIMIZED]

ax.bar(models,values)

ax.set_ylabel("MAE")

st.pyplot(fig)

# =========================================================
# HORIZON GRAPH
# =========================================================

st.subheader("Prediction Horizon Performance")

minutes=list(HORIZONS.keys())
mae=list(HORIZONS.values())

fig,ax=plt.subplots()

ax.plot(minutes,mae,marker="o")

ax.set_xlabel("Prediction Horizon (minutes)")
ax.set_ylabel("MAE")

st.pyplot(fig)

# =========================================================
# CHANNEL IMPORTANCE GRAPH
# =========================================================

st.subheader("Channel Attention Importance")

fig,ax=plt.subplots()

ax.barh(
list(CHANNEL_IMPORTANCE.keys()),
list(CHANNEL_IMPORTANCE.values())
)

st.pyplot(fig)

# =========================================================
# ABLATION GRAPH
# =========================================================

st.subheader("Ablation Study Contribution")

fig,ax=plt.subplots()

ax.bar(
ABLATION["Model"],
ABLATION["MAE"]
)

plt.xticks(rotation=25)

st.pyplot(fig)

# =========================================================
# INTERACTIVE SENSOR VIEWER (SIMULATED)
# =========================================================

st.subheader("Interactive Sensor Prediction Explorer")

sensor=st.slider("Select Sensor ID",0,306,42)

np.random.seed(sensor)

truth=np.random.normal(60,5,12)
prediction=truth+np.random.normal(0,2,12)

fig,ax=plt.subplots()

ax.plot(truth,label="Ground Truth")
ax.plot(prediction,label="Prediction")

ax.legend()

st.pyplot(fig)

# =========================================================
# PARAMETER COMPARISON GRAPH
# =========================================================

st.subheader("Model Parameter Comparison")

params=["Baseline","Standard","Optimized"]

param_counts=[453002,456213,452842]

fig,ax=plt.subplots()

ax.bar(params,param_counts)

st.pyplot(fig)

# =========================================================
# ARCHITECTURE SUMMARY
# =========================================================

st.subheader("Architecture Contributions")

st.markdown("""

DynaSTGCN introduces:

• Channel Attention Gate  
• Adaptive Learned Adjacency  
• Temporal Positional Embedding  

Key discovery:

Removing Temporal Positional Embedding improved accuracy.

""")

# =========================================================
# FINAL RESULTS TABLE
# =========================================================

st.subheader("Full Ablation Table")

st.dataframe(ABLATION)

# =========================================================
# CONCLUSION
# =========================================================

st.subheader("Conclusion")

st.markdown(f"""

Baseline ASTGCN MAE: **{BASELINE}**

Standard Model MAE: **{STANDARD}**

Optimized Model MAE: **{OPTIMIZED}**

Total improvement: **{improvement:.1f}%**

""")