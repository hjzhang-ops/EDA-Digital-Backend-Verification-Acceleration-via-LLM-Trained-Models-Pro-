
import streamlit as st
import json, os, pandas as pd

st.title("EDA LLM Verification Dashboard")
st.write("Inspect the latest predictions and extracted features.")

pred_file = "results/predictions/latest.json"
if os.path.exists(pred_file):
    with open(pred_file) as f:
        data = json.load(f)
    st.subheader("Prediction")
    st.json(data["prediction"])
    st.subheader("Extracted Features")
    st.json(data["structured"].get("features", {}))
else:
    st.info("Run the pipeline first to generate predictions.")
