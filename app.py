import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Insurance Manager Dashboard", page_icon="🏥", layout="wide")

st.title("🏥 Insurance Manager Dashboard — Dialysis & Diabetes Focus")
st.markdown("Monitor insurance claims, costs, and risk trends for chronic conditions like **Dialysis** and **Diabetes**.")

st.markdown("---")
st.subheader("📊 Navigation Guide")

st.markdown("""
- **Daily View:** Operational alerts (pending vs. processed claims, frauds)
- **Weekly Performance:** Efficiency by provider and payer
- **Monthly Overview:** PMPM, denial rate, condition cost share
- **Predictive Insights:** Forecast future claim cost using Random Forest
""")

data_path = "data/cleaned_claims_full.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.success(f"✅ Data Loaded — {len(df):,} records available for analysis")
    st.dataframe(df.head())
else:
    st.warning("⚠️ Please run `scripts/data_cleaning.py` first to create cleaned_claims_full.csv.")
