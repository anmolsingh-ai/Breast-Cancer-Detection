import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Prediction System")
st.write("Powered by FastAPI + ML Ensemble Model")

# ------------------------
# Mode Selection
# ------------------------
mode = st.radio("Choose Mode", ["Single Prediction", "Batch Prediction (CSV Upload)"])

# ------------------------
# Single Prediction UI
# ------------------------
if mode == "Single Prediction":
    st.subheader("🔢 Enter Patient Features")

    # You can rename these later to real feature names
    f1 = st.number_input("Feature 1")
    f2 = st.number_input("Feature 2")
    f3 = st.number_input("Feature 3")
    f4 = st.number_input("Feature 4")
    f5 = st.number_input("Feature 5")
    f6 = st.number_input("Feature 6")
    f7 = st.number_input("Feature 7")
    f8 = st.number_input("Feature 8")
    f9 = st.number_input("Feature 9")
    f10 = st.number_input("Feature 10")
    f11 = st.number_input("Feature 11")
    f12 = st.number_input("Feature 12")
    f13 = st.number_input("Feature 13")
    f14 = st.number_input("Feature 14")
    f15 = st.number_input("Feature 15")
    f16 = st.number_input("Feature 16")
    f17 = st.number_input("Feature 17")
    f18 = st.number_input("Feature 18")
    f19 = st.number_input("Feature 19")
    f20 = st.number_input("Feature 20")
    f21 = st.number_input("Feature 21")
    f22 = st.number_input("Feature 22")
    f23 = st.number_input("Feature 23")
    f24 = st.number_input("Feature 24")
    f25 = st.number_input("Feature 25")
    f26 = st.number_input("Feature 26")
    f27 = st.number_input("Feature 27")
    f28 = st.number_input("Feature 28")
    f29 = st.number_input("Feature 29")
    f30 = st.number_input("Feature 30")
    f31 = st.number_input("Feature 31")
    f32 = st.number_input("Feature 32")
    f33 = st.number_input("Feature 33")
    f34 = st.number_input("Feature 34")
    f35 = st.number_input("Feature 35")
    f36 = st.number_input("Feature 36")
    f37 = st.number_input("Feature 37")
    f38 = st.number_input("Feature 38")
    f39 = st.number_input("Feature 39")

    if st.button("🔮 Predict"):
        payload = {
             # This sends the numeric values stored in the variables
            "features" : [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39]
        }

        with st.spinner("Predicting..."):
            res = requests.post(f"{API_URL}/predict", json=payload)
            result = res.json()

        if res.status_code == 200:
            label = "Malignant (Cancer)" if result["prediction"] == 1 else "Benign (No Cancer)"
            st.success(f"🧠 Prediction: **{label}**")
            # Show confidence when available, otherwise display N/A
            prob = result.get('probability')
            if prob is None:
                prob_text = "N/A"
            else:
                try:
                    prob_text = f"{float(prob)*100:.2f}%"
                except Exception:
                    prob_text = "N/A"

            st.metric("Confidence", prob_text)
        else:
            st.error(result["detail"])

# ------------------------
# Batch Prediction UI
# ------------------------
else:
    st.subheader("📂 Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file (no header)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)
        st.write("📊 Preview", df.head())

        if st.button("⚡ Run Predictions"):
            payload = {
                "samples": df.values.tolist()
            }

            with st.spinner("Running predictions..."):
                res = requests.post(f"{API_URL}/predict-batch", json=payload)
                result = res.json()

            if res.status_code == 200:
                df["Prediction"] = result["predictions"]
                df["Probability"] = result["probabilities"]
                st.success("✅ Predictions Completed")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")
            else:
                st.error(result["detail"])
