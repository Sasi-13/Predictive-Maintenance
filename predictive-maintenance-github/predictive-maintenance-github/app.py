"""Streamlit interface for the predictive maintenance demo."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_pipeline.joblib"

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Smart Predictive Maintenance")
st.caption("Machine failure risk estimation using the AI4I 2020 benchmark dataset.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run: `python train_model.py`")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter machine sensor readings")
left, right = st.columns(2)

with left:
    machine_type = st.selectbox("Machine type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=300.0)
    process_temp = st.number_input("Process temperature (K)", min_value=250.0, max_value=400.0, value=310.0)

with right:
    rotational_speed = st.number_input("Rotational speed (rpm)", min_value=0, max_value=5000, value=1500)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=1000.0, value=40.0)
    tool_wear = st.number_input("Tool wear (min)", min_value=0, max_value=500, value=100)

if st.button("Predict machine failure", type="primary"):
    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Estimated failure probability", f"{probability:.1%}")
    c2.metric("Model classification", "Failure" if prediction else "No failure")

    if prediction == 1:
        st.error("⚠️ The model predicts a potential failure. Inspect the machine.")
    else:
        st.success("✅ The model does not predict a failure for these readings.")

    st.progress(min(max(probability, 0.0), 1.0))
    st.info(
        "This is a machine-learning estimate based on a benchmark dataset, "
        "not a guaranteed warning or a substitute for industrial safety systems."
    )

st.divider()
st.subheader("Batch prediction")
uploaded = st.file_uploader(
    "Upload a CSV containing the six input columns listed below",
    type=["csv"],
)
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    required = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    missing = [col for col in required if col not in batch.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
    else:
        batch = batch[required].copy()
        batch["Failure probability"] = model.predict_proba(batch)[:, 1]
        batch["Predicted failure"] = model.predict(batch)
        st.dataframe(batch, use_container_width=True)
        st.download_button(
            "Download predictions",
            batch.to_csv(index=False).encode("utf-8"),
            file_name="machine_failure_predictions.csv",
            mime="text/csv",
        )
