import streamlit as st
import pandas as pd
import joblib
import time

from backend_automl import train_automl

st.set_page_config(layout="wide")
st.title("🚀 Automated ML Pipeline")


# ===============================
# SESSION STORAGE
# ===============================
for key in ["model", "df", "target", "model_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ===============================
# UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)
    st.session_state.target = target

    task = st.radio("Task Type", ["classification", "regression"])


    # ===============================
    # TRAIN
    # ===============================
    if st.button("Train Model"):

        progress = st.progress(0)
        start = time.time()

        for i in range(20):
            time.sleep(0.05)
            progress.progress(i * 5)

        dataset_name = uploaded_file.name.split(".")[0]

        model, score, model_name = train_automl(df, target, task)

        filename = f"{dataset_name}_{model_name}.pkl"

        joblib.dump(model, filename)

        st.session_state.model = model
        st.session_state.model_filename = filename

        end = time.time()

        st.success(f"Training Completed ✅ | {model_name}")
        st.info(f"Score: {score:.4f} | Time: {round(end-start,2)} sec")

        progress.progress(100)


# ===============================
# AFTER TRAINING
# ===============================
if st.session_state.model:

    st.divider()
    st.header("🔮 Prediction & Download")

    model = st.session_state.model
    df = st.session_state.df
    target = st.session_state.target


    # Predict full dataset
    if st.button("Predict on Entire Dataset"):

        preds = model.predict(df.drop(columns=[target]))

        result_df = df.copy()
        result_df["Prediction"] = preds

        st.dataframe(result_df.head())

        st.download_button(
            "⬇ Download Predictions CSV",
            result_df.to_csv(index=False).encode(),
            "predictions.csv"
        )


    # Manual input
    st.subheader("Predict for New Row")

    input_data = {}
    for col in df.drop(columns=[target]).columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            input_data[col] = st.number_input(col)
        else:
            input_data[col] = st.text_input(col)

    if st.button("Predict Row"):
        pred = model.predict(pd.DataFrame([input_data]))[0]
        st.success(f"Prediction: {pred}")


    # Download model
    if st.session_state.model_filename:
        with open(st.session_state.model_filename, "rb") as f:
            st.download_button(
                "⬇ Download Trained Model (.pkl)",
                f,
                file_name=st.session_state.model_filename
            )
