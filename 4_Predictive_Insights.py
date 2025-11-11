import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# PAGE TITLE
# ----------------------------
st.title("🔮 Predictive Insights — Claim Cost Forecasting")

st.markdown("""
Use a trained Random Forest model to predict future claim costs or identify high-risk patients.
Upload new data or use the existing cleaned dataset for predictions.
""")

# ----------------------------
# MODEL & DATA PATHS
# ----------------------------
model_path = "models/random_forest_model.pkl"
data_path = "data/cleaned_claims_full.csv"

# ----------------------------
# DATA LOADING SECTION
# ----------------------------
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.success(f"✅ Cleaned data loaded — {df.shape[0]:,} records")
else:
    st.error("❌ Cleaned dataset not found. Please run `scripts/data_cleaning.py` first.")
    st.stop()

# ----------------------------
# MODEL LOADING SECTION
# ----------------------------
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Random Forest model loaded successfully!")
else:
    st.warning("⚠️ No model found. Training a quick sample model using available data...")

    # Build a simple model on the fly
    feature_cols = [col for col in df.columns if col in ["AGE", "IsDiabetes", "IsDialysis", "TOTAL_CLAIM_COST"]]
    if "TOTAL_CLAIM_COST" in df.columns:
        X = df[["AGE", "IsDiabetes", "IsDialysis"]]
        y = df["TOTAL_CLAIM_COST"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        st.info("🧠 Model trained and saved as random_forest_model.pkl")
    else:
        st.error("❌ TOTAL_CLAIM_COST column not found. Cannot train model.")
        st.stop()

# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.markdown("---")
st.subheader("📊 Run Predictions")

feature_cols = ["AGE", "IsDiabetes", "IsDialysis"]

if all(col in df.columns for col in feature_cols):
    X_pred = df[feature_cols]
    df["PredictedCost"] = model.predict(X_pred)
    st.success("✅ Predictions generated successfully!")

    # ----------------------------
    # CHART 1: DISTRIBUTION
    # ----------------------------
    st.subheader("💰 Predicted Claim Cost Distribution")
    fig1 = px.histogram(
        df,
        x="PredictedCost",
        nbins=40,
        color_discrete_sequence=["#1565C0"],
        title="Distribution of Predicted Claim Costs"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------
    # CHART 2: FEATURE IMPORTANCE
    # ----------------------------
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.subheader("🧠 Feature Importance")
    fig2 = px.bar(
        feat_df,
        x="Feature",
        y="Importance",
        color="Feature",
        title="Top Features Driving Claim Cost Predictions"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # TABLE
    # ----------------------------
    st.markdown("### 📋 Predicted Results Sample")
    st.dataframe(df[["PATIENT", "AGE", "IsDiabetes", "IsDialysis", "PredictedCost"]].head(20))

else:
    st.error("❌ Required features not found (need AGE, IsDiabetes, IsDialysis). Please check your data.")
