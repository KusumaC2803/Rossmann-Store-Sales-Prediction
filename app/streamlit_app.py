import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
page_title="Rossmann Sales Forecasting",
page_icon="📊",
layout="wide",
)

st.title("📊 Rossmann Store Sales Prediction")
st.write(
"An interactive portfolio project exploring retail sales forecasting, "
"promotion patterns, and model evaluation."
)

st.info(
"This dashboard presents the project's existing validation results. "
"It does not train the model or generate new predictions."
)

st.subheader("Model Validation Results")

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", "725.94")
col2.metric("Root Mean Squared Error (RMSE)", "1,130.36")

st.caption(
"Metrics are from chronological recursive validation. "
"They are not measurements against the competition's unseen test labels."
)

st.subheader("Project Details")

st.markdown(
"""
- **Model:** Random Forest Regressor
- **Problem:** Daily retail sales forecasting
- **Features:** Calendar, promotion, competition, lag, and rolling-sales features
- **Validation:** Chronological recursive forecasting
"""
)

st.markdown(
"[View the source code on GitHub](https://github.com/KusumaC2803/Rossmann-Store-Sales-Prediction)"
)

st.subheader("Validation Results")

results_path = (
Path(**file**).resolve().parent.parent
/ "outputs"
/ "baseline_comparison.csv"
)

if results_path.exists():
results = pd.read_csv(results_path)
st.dataframe(results, use_container_width=True)
else:
st.warning(
"The validation CSV is not included in this deployment. "
"The reported metrics above come from the project's recorded validation run."
)
