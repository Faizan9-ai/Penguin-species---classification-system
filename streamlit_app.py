
"""
streamlit_app.py
Streamlit interface for the optimized penguin species classifier.
Place optimized_penguin_model.pkl in the same directory before running:
    streamlit run streamlit_app.py
"""

import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
import os

st.set_page_config(page_title="Penguin Species Predictor", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "optimized_penguin_model.pkl")

st.title(" Penguin Species Predictor")
st.write("Predict the species of penguins (Adélie, Chinstrap, Gentoo) using physical measurements and metadata.")

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    return model

model = load_model()

if model is None:
    st.error(f"Model file not found: {MODEL_PATH}. Please place the trained model in the app folder.")
    st.stop()

# --- Sidebar: batch upload or single input ---
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Mode", ("Single input", "Batch CSV"))

# common inputs for numeric ranges
min_max = {
    "culmen_length_mm": (20.0, 60.0), 
    "culmen_depth_mm": (8.0, 22.0),
    "flipper_length_mm": (170, 240),
    "body_mass_g": (2000, 7000)
}

def compute_derived(df):
    # avoid division by zero
    df['culmen_ratio'] = df.apply(lambda r: r['culmen_length_mm']/r['culmen_depth_mm'] if r['culmen_depth_mm'] and r['culmen_depth_mm']>0 else 0, axis=1)
    df['body_mass_kg'] = df['body_mass_g'] / 1000.0
    return df

if mode == "Single input":
    st.subheader("Enter features for a single penguin")

    col1, col2 = st.columns(2)
    with col1:
        culmen_length_mm = st.number_input("Culmen length (mm)", value=45.0, format="%.2f",
                                           min_value=min_max['culmen_length_mm'][0], max_value=min_max['culmen_length_mm'][1])
        culmen_depth_mm = st.number_input("Culmen depth (mm)", value=14.0, format="%.2f",
                                          min_value=min_max['culmen_depth_mm'][0], max_value=min_max['culmen_depth_mm'][1])
        flipper_length_mm = st.number_input("Flipper length (mm)", value=200, step=1,
                                            min_value=min_max['flipper_length_mm'][0], max_value=min_max['flipper_length_mm'][1])
    with col2:
        body_mass_g = st.number_input("Body mass (g)", value=4500, step=10,
                                      min_value=min_max['body_mass_g'][0], max_value=min_max['body_mass_g'][1])
        island = st.selectbox("Island", options=["Biscoe", "Dream", "Torgersen"])
        sex = st.selectbox("Sex", options=["male", "female", "Unknown"])

    if st.button("Predict species"):
        single = pd.DataFrame([{
            'culmen_length_mm': culmen_length_mm,
            'culmen_depth_mm': culmen_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'island': island,
            'sex': sex
        }])

        single = compute_derived(single)

        # Make prediction
        pred = model.predict(single)[0]
        proba = None
        try:
            proba = model.predict_proba(single)[0]
            classes = model.classes_
            prob_df = pd.DataFrame({'species': classes, 'probability': proba})
            prob_df = prob_df.sort_values('probability', ascending=False).reset_index(drop=True)
        except Exception:
            prob_df = None

        st.success(f"Predicted species: **{pred}**")
        if prob_df is not None:
            st.subheader("Prediction probabilities")
            st.table(prob_df.style.format({'probability': "{:.3f}"}))

        # If model has feature importances, show top features
        try:
            clf = model.named_steps.get('classifier', None) or model
            fi = getattr(clf, "feature_importances_", None)
            if fi is not None:
                # get feature names from preprocessor
                pre = model.named_steps['preprocessor']
                num_feats = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','culmen_ratio','body_mass_kg']
                cat_feats = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['island','sex']))
                feat_names = num_feats + cat_feats
                fi_series = pd.Series(fi, index=feat_names).sort_values(ascending=False).head(10)
                st.subheader("Top feature importances")
                st.bar_chart(fi_series)
        except Exception:
            pass

elif mode == "Batch CSV":
    st.subheader("Upload a CSV file for batch predictions")
    st.markdown("CSV must include columns: `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`, `island`, `sex`.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        missing = set(['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','island','sex']) - set(df.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            df = compute_derived(df)
            preds = model.predict(df)
            try:
                prob = model.predict_proba(df)
                prob_df = pd.DataFrame(prob, columns=[f"prob_{c}" for c in model.classes_])
                out = pd.concat([df.reset_index(drop=True), pd.Series(preds, name='predicted_species'), prob_df], axis=1)
            except Exception:
                out = df.copy()
                out['predicted_species'] = preds

            st.success("Batch predictions completed")
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download results CSV", data=csv, file_name="penguin_predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: If you don't have the model file yet, run the training pipeline to produce `optimized_penguin_model.pkl` and place it in this folder.")
