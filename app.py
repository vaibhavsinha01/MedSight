"""Streamlit interface for the three local medical models."""
import base64
import asyncio
import json
import streamlit as st
from agents.agents import AVAILABLE_IMAGE_MODELS, AVAILABLE_TABULAR_MODELS
from agents.research import ResearchAgent
from models.models import ModelNotAvailableError, ModelRegistry, predict_image, predict_tabular

st.set_page_config(page_title="Medical Model Studio", page_icon="🩺", layout="wide")
st.title("🩺 Medical Model Studio")
st.caption("Chest X-ray, heart disease and lifestyle risk predictions with local explainability.")
st.info("For research and education only. Model outputs are not a diagnosis or treatment recommendation.")

with st.sidebar:
    st.subheader("Model status")
    if st.button("Refresh status"):
        st.rerun()
    for name, info in ModelRegistry.status().items():
        st.write(("✅" if info["weights_found_on_disk"] else "⚠️") + f" **{name}**")
        if not info["weights_found_on_disk"]:
            st.caption(f"Missing `{info['weight_file']}` in weights/")

image_tab, heart_tab, risk_tab, research_tab = st.tabs(["Chest X-ray", "Heart disease", "Risk", "Research agent"])

def show_result(result):
    st.success(f"**{result.predicted_label}** · {result.confidence:.1%} model confidence")
    st.bar_chart(result.class_probabilities)
    exp = result.explanation
    st.caption("Explanation methods: " + ", ".join(exp.get("methods", [])))
    for key, title in (("gradcam_png_base64", "Grad-CAM"), ("saliency_png_base64", "Saliency map"),
                       ("occlusion_sensitivity_png_base64", "Occlusion sensitivity")):
        if exp.get(key):
            st.image(base64.b64decode(exp[key]), caption=title, use_container_width=True)
    if exp.get("top_features"):
        st.subheader("Local feature attributions")
        st.dataframe(exp["top_features"], use_container_width=True, hide_index=True)
    for warning in result.warnings:
        st.warning(warning)

with image_tab:
    st.subheader("Chest X-ray")
    file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
    if file:
        st.image(file, width=360)
    if st.button("Analyze X-ray", type="primary"):
        if not file:
            st.error("Choose an image first.")
        else:
            try:
                with st.spinner("Predicting and computing explanations…"):
                    show_result(predict_image("chest_xray", file.getvalue()))
            except ModelNotAvailableError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with heart_tab:
    st.subheader("Heart disease model")
    with st.form("heart_form"):
        c1, c2 = st.columns(2)
        with c1:
            features = {"Respiratory_Rate": st.number_input("Respiratory rate", value=18.),
                        "Oxygen_Saturation": st.number_input("Oxygen saturation (%)", value=97.),
                        "O2_Scale": st.number_input("O₂ scale", value=1.),
                        "Systolic_BP": st.number_input("Systolic blood pressure", value=120.)}
        with c2:
            features.update({"Heart_Rate": st.number_input("Heart rate", value=80.),
                             "Temperature": st.number_input("Temperature (°C)", value=36.8),
                             "Consciousness": st.number_input("Consciousness score", value=0.),
                             "On_Oxygen": st.selectbox("On oxygen", [0, 1])})
        submitted = st.form_submit_button("Estimate heart disease risk", type="primary")
    if submitted:
        try:
            with st.spinner("Predicting and computing SHAP/LIME explanations…"):
                show_result(predict_tabular("heart_disease", features))
        except ModelNotAvailableError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with risk_tab:
    st.subheader("Lifestyle risk model")
    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            features = {"age": st.number_input("Age", value=35.), "weight": st.number_input("Weight (kg)", value=70.),
                        "height": st.number_input("Height (cm)", value=170.),
                        "exercise": st.selectbox("Exercise", ["low", "medium", "high"]),
                        "sleep": st.number_input("Sleep (hours)", value=7.),
                        "sugar_intake": st.selectbox("Sugar intake", ["low", "medium", "high"])}
        with c2:
            features.update({"smoking": st.selectbox("Smoking", ["no", "yes"]),
                             "alcohol": st.selectbox("Alcohol", ["no", "yes"]),
                             "married": st.selectbox("Married", ["no", "yes"]),
                             "profession": st.selectbox("Profession", ["office_worker", "teacher", "artist", "farmer", "driver", "engineer", "student", "other"]),
                             "bmi": st.number_input("BMI", value=24.)})
        submitted = st.form_submit_button("Estimate lifestyle risk", type="primary")
    if submitted:
        try:
            with st.spinner("Predicting and computing SHAP/LIME explanations…"):
                show_result(predict_tabular("risk", features))
        except ModelNotAvailableError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with research_tab:
    st.subheader("Evidence search")
    st.caption("Searches PubMed abstracts, then uses Gemini to summarize only those sources.")
    question = st.text_area("Research question", placeholder="e.g. What does recent evidence say about exercise and cardiovascular risk?")
    consent = st.checkbox("I agree to send this question to PubMed and Gemini for retrieval and summarization.")
    if st.button("Search and summarize", type="primary", disabled=not consent):
        try:
            with st.spinner("Retrieving PubMed abstracts and preparing a grounded summary…"):
                result = asyncio.run(ResearchAgent().run(question))
            st.markdown(result["answer"])
            if result["citations"]:
                st.subheader("Retrieved sources")
                for i, paper in enumerate(result["citations"], 1):
                    st.markdown(f"[{i}] [{paper['title']}]({paper['url']}) · {paper['journal']} · {paper['year']}")
        except Exception as exc:
            st.error(f"Research failed: {exc}")
