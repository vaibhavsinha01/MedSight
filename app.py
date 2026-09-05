# app.py
"""
Streamlit UI for the Medical Agentic AI System.

This talks DIRECTLY to the existing Python backend (agents/graph.py,
agents/agents.py, models/models.py, rag/rag.py) — there is no separate
FastAPI server or frontend/ HTML involved. Run it with:

    streamlit run app.py

Tabs:
  - Chat      : full LangGraph multi-agent workflow (supervisor -> vision/research -> synthesis -> eval)
  - Predict   : direct model inference (image or tabular) + Grad-CAM / feature-importance XAI
  - Research  : direct Agentic RAG (PubMed/PMC) query, bypassing the graph

Research and educational use only. Not intended for medical diagnosis or treatment.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import streamlit as st

# --------------------------------------------------------------------------
# Backend imports — same modules api.py uses, just called in-process instead
# of over HTTP.
# --------------------------------------------------------------------------
from agents.agents import AVAILABLE_IMAGE_MODELS, AVAILABLE_TABULAR_MODELS
from agents.graph import MedicalAgentOrchestrator
from models.models import ModelNotAvailableError, ModelRegistry, predict_image, predict_tabular
from rag.rag import agentic_rag_query

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Medical Agentic AI", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def run_async(coro):
    """Bridge async backend calls (orchestrator/RAG are all `async def`) into
    Streamlit's synchronous script execution model."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback if an event loop is already running in this context.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> MedicalAgentOrchestrator:
    """Built once per Streamlit session process — mirrors api.py's singleton pattern."""
    return MedicalAgentOrchestrator()


def render_prediction(result_dict: dict) -> None:
    st.json({
        "model": result_dict["model_name"],
        "task": result_dict["task"],
        "label": result_dict["predicted_label"],
        "confidence": result_dict["confidence"],
        "class_probabilities": result_dict["class_probabilities"],
    })
    if result_dict.get("warnings"):
        for w in result_dict["warnings"]:
            st.warning(w)
    explanation = result_dict.get("explanation", {})
    if explanation.get("gradcam_png_base64"):
        st.image(
            base64.b64decode(explanation["gradcam_png_base64"]),
            caption=f"Grad-CAM (conv layer: {explanation.get('conv_layer_used', 'n/a')})",
            use_container_width=True,
        )
    if explanation.get("top_features"):
        st.caption("Top contributing features (XGBoost feature importance)")
        st.table(explanation["top_features"])


st.title("🩺 Medical Agentic AI System")
st.caption(
    "Multi-agent medical assistant — trained ML/DL models + Agentic RAG + LangGraph orchestration. "
    "**Research and educational use only. Not a substitute for professional medical advice.**"
)

with st.sidebar:
    st.subheader("System status")
    if st.button("Refresh model status"):
        st.cache_data.clear()
    status = ModelRegistry.status()
    for name, info in status.items():
        icon = "✅" if info["weights_found_on_disk"] else "⚠️"
        st.write(f"{icon} `{name}` — {'weights found' if info['weights_found_on_disk'] else 'weights missing'}")
        if info.get("last_error"):
            st.caption(f"Last error: {info['last_error']}")

tab_chat, tab_predict, tab_research = st.tabs(["💬 Chat", "🔬 Predict", "📚 Research"])

# --------------------------------------------------------------------------
# Chat tab — full LangGraph multi-agent workflow (mirrors POST /chat)
# --------------------------------------------------------------------------
with tab_chat:
    st.header("Multi-Agent Chat")
    st.caption(
        "Routed through the Supervisor Agent -> Vision/Research agents -> synthesis -> Evaluation "
        "Agent, exactly like the `/chat` API endpoint."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    col_img, col_tab = st.columns(2)
    with col_img:
        chat_image = st.file_uploader(
            "Attach an image (optional)", type=["png", "jpg", "jpeg"], key="chat_image_uploader"
        )
        chat_image_model = st.selectbox(
            "Image model (optional hint)", options=["(auto)"] + AVAILABLE_IMAGE_MODELS, key="chat_image_model"
        )
    with col_tab:
        chat_tabular_json = st.text_area(
            "Attach tabular vitals/lifestyle features as JSON (optional)",
            height=100,
            placeholder='{"Systolic_BP": 120, "Heart_Rate": 80, ...}',
            key="chat_tabular_json",
        )
        chat_tabular_model = st.selectbox(
            "Tabular model (optional hint)", options=["(auto)"] + AVAILABLE_TABULAR_MODELS, key="chat_tabular_model"
        )

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_message = st.chat_input("Ask a medical question, or describe the attached image/data...")

    if user_message:
        st.session_state["chat_history"].append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        image_bytes = chat_image.read() if chat_image is not None else None
        tabular_features = None
        if chat_tabular_json.strip():
            try:
                tabular_features = json.loads(chat_tabular_json)
            except json.JSONDecodeError:
                st.error("Tabular features JSON is invalid — ignoring it for this turn.")

        orch = get_orchestrator()
        with st.chat_message("assistant"):
            with st.spinner("Running supervisor -> agents -> evaluation..."):
                try:
                    state = run_async(orch.run(
                        user_query=user_message,
                        image_bytes=image_bytes,
                        image_model_hint=None if chat_image_model == "(auto)" else chat_image_model,
                        tabular_features=tabular_features,
                        tabular_model_hint=None if chat_tabular_model == "(auto)" else chat_tabular_model,
                    ))
                    answer = state.get("final_answer", "(no answer produced)")
                    st.markdown(answer)
                    st.caption(f"Route: `{state.get('route')}` — {state.get('route_rationale', '')}")

                    if state.get("prediction"):
                        with st.expander("Model prediction details"):
                            render_prediction(state["prediction"])
                    if state.get("research"):
                        with st.expander("Research / citations"):
                            for cit in state["research"].get("citations", []):
                                st.write(cit)
                    if state.get("evaluation"):
                        with st.expander("Evaluation (LLM-as-judge + metrics)"):
                            st.json(state["evaluation"])
                    if state.get("a2a_trace"):
                        with st.expander("Agent-to-agent trace"):
                            st.json(state["a2a_trace"])

                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Agent workflow failed: {e}")

# --------------------------------------------------------------------------
# Predict tab — direct model inference, bypassing the graph (mirrors /predict)
# --------------------------------------------------------------------------
with tab_predict:
    st.header("Direct Model Prediction")
    predict_subtabs = st.tabs(["🖼️ Image", "📋 Tabular"])

    # ---- Image prediction ----
    with predict_subtabs[0]:
        st.subheader("Image Prediction")
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="predict_image_uploader")
        model_name = st.selectbox("Select image model", options=AVAILABLE_IMAGE_MODELS, key="predict_image_model")
        with_gradcam = st.checkbox("Generate Grad-CAM", value=True, key="predict_gradcam_checkbox")

        if image_file is not None:
            st.image(image_file, caption="Uploaded image", width=300)

        if st.button("Run Prediction", key="run_image_prediction"):
            if not image_file:
                st.error("Please upload an image.")
            else:
                image_bytes = image_file.getvalue()
                try:
                    with st.spinner(f"Running {model_name}..."):
                        result = predict_image(model_name, image_bytes, with_gradcam=with_gradcam)
                    st.success("Prediction completed")
                    render_prediction({
                        "model_name": result.model_name,
                        "task": result.task,
                        "predicted_label": result.predicted_label,
                        "confidence": round(result.confidence, 4),
                        "class_probabilities": {k: round(v, 4) for k, v in result.class_probabilities.items()},
                        "warnings": result.warnings,
                        "explanation": result.explanation,
                    })
                except ModelNotAvailableError as e:
                    st.error(f"Model not available: {e}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    # ---- Tabular prediction ----
    with predict_subtabs[1]:
        st.subheader("Tabular Prediction")
        model_name_tab = st.selectbox("Select tabular model", options=AVAILABLE_TABULAR_MODELS, key="predict_tabular_model")

        # Simple guided form so users don't need to hand-write JSON, plus a raw-JSON
        # override for power users / anything the form doesn't cover.
        features: dict = {}
        if model_name_tab == "vitals_risk":
            c1, c2 = st.columns(2)
            with c1:
                features["Respiratory_Rate"] = st.number_input("Respiratory Rate", value=18.0)
                features["Oxygen_Saturation"] = st.number_input("Oxygen Saturation (%)", value=97.0)
                features["O2_Scale"] = st.number_input("O2 Scale", value=1.0)
                features["Systolic_BP"] = st.number_input("Systolic BP", value=120.0)
            with c2:
                features["Heart_Rate"] = st.number_input("Heart Rate", value=80.0)
                features["Temperature"] = st.number_input("Temperature (°C)", value=36.8)
                features["Consciousness"] = st.number_input("Consciousness (0=alert)", value=0.0)
                features["On_Oxygen"] = st.selectbox("On Oxygen", options=[0, 1])
        else:  # lifestyle_risk
            c1, c2 = st.columns(2)
            with c1:
                features["age"] = st.number_input("Age", value=35.0)
                features["weight"] = st.number_input("Weight (kg)", value=70.0)
                features["height"] = st.number_input("Height (cm)", value=170.0)
                features["exercise"] = st.selectbox("Exercise level", options=["low", "medium", "high"])
                features["sleep"] = st.number_input("Sleep (hours)", value=7.0)
                features["sugar_intake"] = st.selectbox("Sugar intake", options=["low", "medium", "high"])
            with c2:
                features["smoking"] = st.selectbox("Smoking", options=["no", "yes"])
                features["alcohol"] = st.selectbox("Alcohol", options=["no", "yes"])
                features["married"] = st.selectbox("Married", options=["no", "yes"])
                features["profession"] = st.selectbox(
                    "Profession",
                    options=["office_worker", "teacher", "artist", "farmer", "driver", "engineer", "student", "other"],
                )
                features["bmi"] = st.number_input("BMI", value=24.0)

        with st.expander("Advanced: override with raw JSON"):
            raw_json_override = st.text_area("Feature JSON (overrides the form above if non-empty)", height=120)

        if st.button("Run Tabular Prediction", key="run_tabular_prediction"):
            try:
                final_features = json.loads(raw_json_override) if raw_json_override.strip() else features
                with st.spinner(f"Running {model_name_tab}..."):
                    result = predict_tabular(model_name_tab, final_features)
                st.success("Prediction completed")
                render_prediction({
                    "model_name": result.model_name,
                    "task": result.task,
                    "predicted_label": result.predicted_label,
                    "confidence": round(result.confidence, 4),
                    "class_probabilities": {k: round(v, 4) for k, v in result.class_probabilities.items()},
                    "warnings": result.warnings,
                    "explanation": result.explanation,
                })
            except json.JSONDecodeError:
                st.error("Invalid JSON in the advanced override box.")
            except ModelNotAvailableError as e:
                st.error(f"Model not available: {e}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# --------------------------------------------------------------------------
# Research tab — direct Agentic RAG query, bypassing the graph (mirrors /research)
# --------------------------------------------------------------------------
with tab_research:
    st.header("Agentic RAG Query (PubMed / PMC)")
    query = st.text_area("Enter your research query", height=120, key="research_query")
    max_rounds = st.number_input("Max retrieval rounds", min_value=1, max_value=10, value=2, step=1, key="research_max_rounds")

    if st.button("Run Research", key="run_research"):
        if not query.strip():
            st.error("Please provide a query.")
        else:
            orch = get_orchestrator()
            try:
                with st.spinner("Retrieving and synthesizing evidence..."):
                    result = run_async(agentic_rag_query(query, llm_call=orch.llm.complete, max_rounds=int(max_rounds)))
                st.success("Research completed")
                st.subheader("Answer")
                st.write(result.answer)
                st.caption(
                    f"Retrieval rounds: {result.retrieval_rounds} | "
                    f"Sufficient evidence found: {result.sufficient_evidence_found}"
                )
                if result.citations:
                    st.subheader("Citations")
                    for cit in result.citations:
                        st.write(cit)
            except Exception as e:
                st.error(f"Error during research: {e}")

# End of app.py