"""
models/models.py
=================
Loads YOUR actual trained models (produced by Medical_research_agent.ipynb) from disk
and exposes clean prediction functions with Grad-CAM explainability for the CNN models.

This module does NOT reconstruct architectures and does NOT train stub/fallback models.
It loads exactly what you saved:

  Tabular (XGBoost, joblib.dump):
    - heart_disease     <- xgboost_heart_disease.joblib   (3-class: Low/Medium/High)
    - risk  <- xgboost_risk.joblib             (binary: Low/High)

  Vision (Keras model.save / .keras or .h5):
    - chest_xray            <- chest_xray_model.h5                        (NORMAL/PNEUMONIA/COVID)

Drop the corresponding files into WEIGHTS_DIR (see weights/README.md). If a file is
missing, the loader raises ModelNotAvailableError when the model is used.

Grad-CAM is implemented generically against whatever real model gets loaded: it walks
model.layers (recursing into nested submodels such as an EfficientNet backbone) to find
the last Conv layer, then builds the gradient tap graph from that layer's OWNING
submodel's own already-connected functional graph, replaying any outer layers on top
inside the GradientTape. This avoids Keras 3's tendency to produce a disconnected graph
when you naively do `Model(inputs=model.inputs, outputs=[nested_layer.output, model.output])`.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("medagent.models")

WEIGHTS_DIR = os.environ.get(
    "WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "..", "weights")
)

# Lazy heavy imports so importing this module (and booting the API) doesn't require
# TensorFlow/XGBoost to be installed unless a model is actually used.
_tf = None
_joblib = None


def _tf_lib():
    global _tf
    if _tf is None:
        import tensorflow as tf  # noqa
        _tf = tf
    return _tf


def _joblib_lib():
    global _joblib
    if _joblib is None:
        import joblib  # noqa
        _joblib = joblib
    return _joblib


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested model's weight file is not present in WEIGHTS_DIR, or fails to load."""


def _tabular_explanations(model, row: np.ndarray, features: list[str], probabilities: np.ndarray):
    """Return local SHAP and LIME attributions; either method can fail independently."""
    warnings, values = [], {}
    try:
        import shap
        explanation = shap.TreeExplainer(model).shap_values(row.reshape(1, -1))
        target = int(np.argmax(probabilities))
        if isinstance(explanation, list):
            explanation = explanation[target]
        explanation = np.asarray(explanation)
        if explanation.ndim == 3:
            explanation = explanation[0, :, target]
        elif explanation.ndim == 2:
            explanation = explanation[0]
        explanation = explanation.reshape(-1)
        values["SHAP"] = explanation
    except Exception as exc:
        warnings.append(f"SHAP unavailable: {exc}")
    try:
        from lime.lime_tabular import LimeTabularExplainer
        scale = np.maximum(np.abs(row), 1.0) * 0.15
        background = row + np.random.default_rng(7).normal(size=(128, len(row))) * scale
        explainer = LimeTabularExplainer(background, feature_names=features, class_names=[str(i) for i in range(len(probabilities))],
                                         mode="classification", discretize_continuous=False, random_state=7)
        local = explainer.explain_instance(row, model.predict_proba, num_features=min(6, len(features)), num_samples=256)
        values["LIME"] = np.zeros(len(features))
        for index, weight in local.as_map().get(int(np.argmax(probabilities)), []):
            values["LIME"][index] = weight
    except Exception as exc:
        warnings.append(f"LIME unavailable: {exc}")
    ranked = []
    for method, attribution in values.items():
        ranked.extend({"feature": name, "importance": round(float(score), 5), "method": method}
                      for name, score in zip(features, attribution))
    return sorted(ranked, key=lambda item: abs(item["importance"]), reverse=True)[:10], warnings


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class PredictionResult:
    model_name: str
    task: str
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    modality: str  # "tabular" | "image"
    explanation: dict[str, Any] = field(default_factory=dict)  # gradcam_png_base64, top_features, etc.
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Tabular models: XGBoost (joblib)
# --------------------------------------------------------------------------

class VitalsRiskModel:
    """Loads xgboost_heart_disease.joblib — trained on Health_Risk_Dataset.csv (notebook cells 7-10)."""

    NAME = "heart_disease"
    WEIGHT_FILE = "xgboost_heart_disease.joblib"
    FEATURES = [
        "Respiratory_Rate", "Oxygen_Saturation", "O2_Scale",
        "Systolic_BP", "Heart_Rate", "Temperature", "Consciousness", "On_Oxygen",
    ]
    CLASSES = ["Low", "Medium", "High"]

    def __init__(self):
        self.model = self._load()

    def _load(self):
        path = os.path.join(WEIGHTS_DIR, self.WEIGHT_FILE)
        if not os.path.exists(path):
            raise ModelNotAvailableError(
                f"'{self.WEIGHT_FILE}' not found in {WEIGHTS_DIR}. "
                f"Copy your trained joblib file there (see weights/README.md)."
            )
        model = _joblib_lib().load(path)
        logger.info(f"[{self.NAME}] loaded {path}")
        return model

    def predict(self, features: dict[str, float]) -> PredictionResult:
        x = np.array([[float(features.get(f, 0.0)) for f in self.FEATURES]])
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        n_classes = len(proba)
        classes = self.CLASSES[:n_classes] if n_classes <= len(self.CLASSES) else \
            self.CLASSES + [f"class_{i}" for i in range(len(self.CLASSES), n_classes)]
        probs = {c: float(p) for c, p in zip(classes, proba)}

        warnings: list[str] = []
        top_features, xai_warnings = _tabular_explanations(self.model, x[0], self.FEATURES, proba)
        warnings.extend(xai_warnings)

        return PredictionResult(
            model_name=self.NAME,
            task="Vital-signs risk stratification",
            predicted_label=classes[idx],
            confidence=float(proba[idx]),
            class_probabilities=probs,
            modality="tabular",
            explanation={"top_features": top_features, "methods": ["SHAP", "LIME"]},
            warnings=warnings,
        )


class LifestyleRiskModel:
    """Loads the paired lifestyle risk model and StandardScaler artifacts."""

    NAME = "risk"
    WEIGHT_FILE = "xgboost_risk.joblib"
    FEATURES = ["age", "weight", "height", "exercise", "sleep", "sugar_intake",
                "smoking", "alcohol", "married", "profession", "bmi"]
    CLASSES = ["Low", "High"]
    EXERCISE_MAP = {"low": 0, "medium": 1, "high": 2}
    SUGAR_MAP = {"low": 0, "medium": 1, "high": 2}
    BOOL_MAP = {"no": 0, "yes": 1}
    PROFESSION_MAP = {"office_worker": 0, "teacher": 1, "artist": 2, "farmer": 3,
                       "driver": 4, "engineer": 5, "student": 6, "other": 7}

    SCALER_FILE = "xgboost_risk_scaler.joblib"

    def __init__(self):
        self.model = self._load()
        self.scaler = self._load_scaler()

    def _load(self):
        path = os.path.join(WEIGHTS_DIR, self.WEIGHT_FILE)
        if not os.path.exists(path):
            raise ModelNotAvailableError(
                f"'{self.WEIGHT_FILE}' not found in {WEIGHTS_DIR}. "
                f"Copy your trained joblib file there (see weights/README.md)."
            )
        model = _joblib_lib().load(path)
        logger.info(f"[{self.NAME}] loaded {path}")
        return model

    def _load_scaler(self):
        path = os.path.join(WEIGHTS_DIR, self.SCALER_FILE)
        if not os.path.exists(path):
            raise ModelNotAvailableError(f"Required scaler '{self.SCALER_FILE}' not found in {WEIGHTS_DIR}.")
        logger.info(f"[{self.NAME}] loaded scaler {path}")
        return _joblib_lib().load(path)

    def _encode(self, raw: dict[str, Any]) -> np.ndarray:
        row = [
            float(raw.get("age", 0)), float(raw.get("weight", 0)), float(raw.get("height", 0)),
            self.EXERCISE_MAP.get(str(raw.get("exercise", "low")).lower(), 0),
            float(raw.get("sleep", 0)),
            self.SUGAR_MAP.get(str(raw.get("sugar_intake", "low")).lower(), 0),
            self.BOOL_MAP.get(str(raw.get("smoking", "no")).lower(), 0),
            self.BOOL_MAP.get(str(raw.get("alcohol", "no")).lower(), 0),
            self.BOOL_MAP.get(str(raw.get("married", "no")).lower(), 0),
            self.PROFESSION_MAP.get(str(raw.get("profession", "other")).lower(), 7),
            float(raw.get("bmi", 0)),
        ]
        return np.array([row], dtype=float)

    def predict(self, raw: dict[str, Any]) -> PredictionResult:
        x = self._encode(raw)
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        n_classes = len(proba)
        classes = self.CLASSES[:n_classes] if n_classes <= len(self.CLASSES) else \
            self.CLASSES + [f"class_{i}" for i in range(len(self.CLASSES), n_classes)]
        probs = {c: float(p) for c, p in zip(classes, proba)}

        warnings: list[str] = []
        top_features, xai_warnings = _tabular_explanations(self.model, x[0], self.FEATURES, proba)
        warnings.extend(xai_warnings)

        return PredictionResult(
            model_name=self.NAME,
            task="Lifestyle-based health risk prediction",
            predicted_label=classes[idx],
            confidence=float(proba[idx]),
            class_probabilities=probs,
            modality="tabular",
            explanation={"top_features": top_features, "methods": ["SHAP", "LIME"]},
            warnings=warnings,
        )


# --------------------------------------------------------------------------
# Vision models: Keras CNNs, loaded from your saved .h5 / .keras files
# --------------------------------------------------------------------------

class _KerasVisionModel:
    """Common loader / predict / Grad-CAM logic. Subclasses only declare metadata
    (weight filename, expected input size, class names) — no architecture code."""

    NAME = "base"
    WEIGHT_FILE = ""
    IMG_SIZE = 224
    CHANNELS = 3
    CLASSES: list[str] = []
    TASK = ""
    RESCALE = True  # divide pixel values by 255 before predict, matching the notebook's preprocessing
    CUSTOM_OBJECTS: dict[str, Any] | None = None  # e.g. custom loss fns registered at save time

    def __init__(self):
        self.model = self._load()
        found = self._find_last_conv_layer()
        self._conv_layer = found[0] if found else None
        self._conv_owner = found[1] if found else None
        # _build_grad_tap_model sets self._grad_tap_uses_full_graph to tell _gradcam whether
        # the tap model's second output IS the final prediction (rebuild strategy) or just
        # the nested owner's own output, requiring outer layers to be replayed on top.
        self._grad_tap_uses_full_graph = False
        self._grad_tap_model = self._build_grad_tap_model()
        if self._conv_layer is None:
            logger.warning(f"[{self.NAME}] no Conv layer found — Grad-CAM will be unavailable for this model.")

    # -- loading -------------------------------------------------------
    def _load(self):
        tf = _tf_lib()
        path = os.path.join(WEIGHTS_DIR, self.WEIGHT_FILE)
        if not os.path.exists(path):
            raise ModelNotAvailableError(
                f"'{self.WEIGHT_FILE}' not found in {WEIGHTS_DIR}. "
                f"Copy your trained Keras model file there (see weights/README.md)."
            )
        kwargs: dict[str, Any] = {"compile": False}
        custom_objects = self.CUSTOM_OBJECTS
        if custom_objects:
            kwargs["custom_objects"] = custom_objects
        try:
            model = tf.keras.models.load_model(path, **kwargs)
        except TypeError:
            # Older/newer TF versions differ on accepting safe_mode kwarg etc. Retry minimal.
            model = tf.keras.models.load_model(path, compile=False)
        logger.info(f"[{self.NAME}] loaded {path}")
        return model

    # -- Grad-CAM graph construction ------------------------------------
    def _find_last_conv_layer(self):
        """Returns (conv_layer, owner_model): owner_model is whichever model object
        (self.model itself, or a nested submodel e.g. an EfficientNet backbone) has its
        OWN connected functional graph containing conv_layer."""
        for layer in reversed(self.model.layers):
            if "Conv" in layer.__class__.__name__:
                return layer, self.model
            if hasattr(layer, "layers"):  # nested functional/Sequential submodel
                for sub in reversed(layer.layers):
                    if "Conv" in sub.__class__.__name__:
                        return sub, layer
        return None

    def _build_grad_tap_model(self):
        """Builds a Model tapping [conv_layer.output, owner.output].

        Three strategies are tried in order, cheapest/most-direct first:
          1. Direct tap of self.model: works when self.model is already a built Functional
             model (has a defined `.output`), whether or not conv_layer sits in a nested
             submodel — as long as self.model.output itself is connected.
          2. Direct tap of the nested owner submodel alone (e.g. an EfficientNet backbone
             built via `EfficientNetB0(...)`, which is Functional and pre-connected at
             construction time even before the outer Sequential/Model wrapping it is
             built). Requires replaying any outer layers after the owner, on top, inside
             the GradientTape (`_grad_tap_uses_full_graph = False`).
          3. Rebuild strategy: replay self.model's own top-level layers on a fresh
             symbolic Input. Needed for plain `Sequential([...])` models loaded from
             .h5/.keras that haven't been called yet, where Keras 3 has no defined
             `.output` at all. Layer objects (and their trained weights) are reused as-is;
             only the graph wiring is reconstructed.
        """
        tf = _tf_lib()
        if self._conv_layer is None or self._conv_owner is None:
            return None

        # Strategy 1: try self.model directly (covers pre-built Functional models,
        # including ones where the conv layer is nested inside a submodel).
        try:
            tap = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self._conv_layer.output, self.model.output],
            )
            self._grad_tap_uses_full_graph = True  # second output IS the final prediction
            return tap
        except Exception as e:
            logger.debug(f"[{self.NAME}] direct self.model tap failed, trying nested-owner tap: {e}")

        # Strategy 2: nested submodel with its own pre-existing connected graph (e.g. an
        # EfficientNet backbone, which is Functional and connected even before the outer
        # Sequential wrapping it has been built/called).
        if self._conv_owner is not self.model:
            try:
                tap = tf.keras.models.Model(
                    inputs=self._conv_owner.inputs,
                    outputs=[self._conv_layer.output, self._conv_owner.output],
                )
                self._grad_tap_uses_full_graph = False  # must replay outer layers in _gradcam
                return tap
            except Exception as e:
                logger.debug(f"[{self.NAME}] nested-owner tap failed, trying rebuild strategy: {e}")

        # Strategy 3: rebuild a connected graph by replaying self.model's own top-level
        # layers on a fresh symbolic Input (handles unbuilt Sequential models).
        try:
            inp = tf.keras.Input(shape=self.model.input_shape[1:])
            x = inp
            conv_out_tensor = None
            for layer in self.model.layers:
                x = layer(x)
                if layer is self._conv_layer:
                    conv_out_tensor = x
            if conv_out_tensor is None:
                logger.warning(
                    f"[{self.NAME}] could not locate conv layer output while rebuilding graph "
                    "(conv layer is likely nested inside a submodel that itself isn't built yet)."
                )
                return None
            self._grad_tap_uses_full_graph = True  # second output IS the final prediction
            return tf.keras.models.Model(inputs=inp, outputs=[conv_out_tensor, x])
        except Exception as e:
            logger.warning(f"[{self.NAME}] could not build Grad-CAM tap model (all strategies failed): {e}")
            return None

    def _outer_layers_after_owner(self) -> list:
        """If the conv layer's owner is a nested submodel, returns self.model's top-level
        layers that come AFTER that submodel, to be replayed on top of the tap model's
        output inside the GradientTape (keeps the gradient path connected end-to-end)."""
        if self._conv_owner is self.model:
            return []
        layers_after = []
        seen_owner = False
        for layer in self.model.layers:
            if seen_owner:
                layers_after.append(layer)
            if layer is self._conv_owner:
                seen_owner = True
        return layers_after

    # -- preprocessing ---------------------------------------------------
    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        tf = _tf_lib()
        img = tf.io.decode_image(image_bytes, channels=self.CHANNELS, expand_animations=False)
        img = tf.image.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = tf.cast(img, tf.float32)
        if self.RESCALE:
            img = img / 255.0
        return img.numpy()[None, ...]

    # -- Grad-CAM ------------------------------------------------------
    def _gradcam(self, x: np.ndarray, class_idx: int) -> np.ndarray | None:
        tf = _tf_lib()
        if self._grad_tap_model is None:
            return None
        try:
            x_tensor = tf.constant(x)
            with tf.GradientTape() as tape:
                conv_out, second_out = self._grad_tap_model(x_tensor, training=False)
                preds = second_out
                if not self._grad_tap_uses_full_graph:
                    # second_out is only the nested owner's own output; replay the
                    # remaining outer layers on top, inside the tape, to reach the
                    # final prediction while keeping the gradient path connected.
                    for layer in self._outer_layers_after_owner():
                        preds = layer(preds, training=False)
                loss = preds[:, 0] if preds.shape[-1] == 1 else preds[:, class_idx]
            grads = tape.gradient(loss, conv_out)
            if grads is None:
                logger.warning(f"[{self.NAME}] Grad-CAM: gradient tape returned None (disconnected graph).")
                return None
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_out_0 = conv_out[0]
            heatmap = conv_out_0 @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
        except Exception as e:
            logger.warning(f"[{self.NAME}] Grad-CAM failed: {type(e).__name__}: {e}")
            return None

    def _overlay_heatmap(self, image_bytes: bytes, heatmap: np.ndarray) -> str:
        import base64
        from PIL import Image
        import matplotlib as mpl

        try:
            jet = mpl.colormaps["jet"]  # matplotlib >= 3.7
        except AttributeError:  # pragma: no cover - older matplotlib fallback
            import matplotlib.cm as cm
            jet = cm.get_cmap("jet")

        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((self.IMG_SIZE, self.IMG_SIZE))
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((self.IMG_SIZE, self.IMG_SIZE))
        heatmap_arr = np.array(heatmap_img)
        colored = jet(heatmap_arr / 255.0)[:, :, :3]
        colored = np.uint8(colored * 255)
        overlay = Image.fromarray(np.uint8(0.55 * np.array(orig) + 0.45 * colored))
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _important_region(self, heatmap: np.ndarray) -> dict[str, float]:
        thresh = heatmap > 0.6
        if not thresh.any():
            thresh = heatmap > (heatmap.max() * 0.8 + 1e-8)
        ys, xs = np.where(thresh)
        if len(xs) == 0:
            return {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        h, w = heatmap.shape
        return {
            "x_min": round(float(xs.min() / w), 3), "x_max": round(float(xs.max() / w), 3),
            "y_min": round(float(ys.min() / h), 3), "y_max": round(float(ys.max() / h), 3),
        }

    # -- predict -----------------------------------------------------------
    def _saliency(self, x: np.ndarray, class_idx: int) -> np.ndarray | None:
        try:
            tf = _tf_lib()
            image = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(image)
                scores = self.model(image, training=False)
                target = scores[:, 0] if scores.shape[-1] == 1 else scores[:, class_idx]
            grad = tape.gradient(target, image)[0]
            heat = tf.reduce_max(tf.abs(grad), axis=-1).numpy()
            return heat / (heat.max() + 1e-8)
        except Exception as exc:
            logger.warning("Saliency failed: %s", exc)
            return None

    def _occlusion(self, x: np.ndarray, class_idx: int) -> np.ndarray | None:
        try:
            tf = _tf_lib()
            base = float(self.model.predict(x, verbose=0)[0][class_idx])
            size, stride = max(16, self.IMG_SIZE // 8), max(8, self.IMG_SIZE // 8)
            samples, coords = [], []
            for y in range(0, self.IMG_SIZE, stride):
                for x0 in range(0, self.IMG_SIZE, stride):
                    y1, x1 = min(y + size, self.IMG_SIZE), min(x0 + size, self.IMG_SIZE)
                    masked = x.copy()
                    masked[:, y:y1, x0:x1, :] = float(np.mean(x))
                    samples.append(masked[0]); coords.append((y, y1, x0, x1))
            probs = self.model.predict(np.asarray(samples), batch_size=32, verbose=0)
            scores = np.asarray([base - float(p[class_idx]) for p in probs])
            heat = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
            for (y, y1, x0, x1), score in zip(coords, scores):
                heat[y:y1, x0:x1] += score
            heat = np.maximum(heat, 0)
            return heat / (heat.max() + 1e-8)
        except Exception as exc:
            logger.warning("Occlusion sensitivity failed: %s", exc)
            return None

    def predict(self, image_bytes: bytes, with_gradcam: bool = True) -> PredictionResult:
        x = self._preprocess(image_bytes)
        raw = self.model.predict(x, verbose=0)[0]

        if np.isscalar(raw) or raw.shape[-1] == 1:
            p1 = float(raw if np.isscalar(raw) else raw[0])
            probs_arr = np.array([1 - p1, p1])
        else:
            probs_arr = raw

        idx = int(np.argmax(probs_arr))
        n = len(probs_arr)
        classes = self.CLASSES[:n] if n <= len(self.CLASSES) else \
            self.CLASSES + [f"class_{i}" for i in range(len(self.CLASSES), n)]
        probs = {c: float(p) for c, p in zip(classes, probs_arr)}

        explanation: dict[str, Any] = {"methods": []}
        warnings: list[str] = []
        if with_gradcam:
            heatmap = self._gradcam(x, idx)
            if heatmap is not None:
                explanation["gradcam_png_base64"] = self._overlay_heatmap(image_bytes, heatmap)
                explanation["important_region"] = self._important_region(heatmap)
                explanation["conv_layer_used"] = self._conv_layer.name if self._conv_layer is not None else None
                explanation["methods"].append("Grad-CAM")
            else:
                warnings.append("Grad-CAM unavailable for this model instance.")
            for method, heat in (("Saliency", self._saliency(x, idx)), ("Occlusion sensitivity", self._occlusion(x, idx))):
                if heat is not None:
                    explanation[method.lower().replace(" ", "_") + "_png_base64"] = self._overlay_heatmap(image_bytes, heat)
                    explanation["methods"].append(method)
                else:
                    warnings.append(f"{method} unavailable for this model instance.")

        return PredictionResult(
            model_name=self.NAME,
            task=self.TASK,
            predicted_label=classes[idx],
            confidence=float(probs_arr[idx]),
            class_probabilities=probs,
            modality="image",
            explanation=explanation,
            warnings=warnings,
        )


class ChestXrayModel(_KerasVisionModel):
    NAME = "chest_xray"
    WEIGHT_FILE = "chest_xray_model.h5"
    IMG_SIZE = 224
    CHANNELS = 1
    CLASSES = ["NORMAL", "PNEUMONIA", "COVID"]
    TASK = "Chest X-ray triage (Normal / Pneumonia / COVID-19)"


# --------------------------------------------------------------------------

class ModelRegistry:
    """Lazily-instantiated singleton registry. A model is only loaded from disk on its
    first actual use; loading errors (missing file, corrupt file) surface as
    ModelNotAvailableError at that point rather than at import time."""

    _instances: dict[str, Any] = {}
    _load_errors: dict[str, str] = {}

    _CATALOG = {
        "heart_disease": VitalsRiskModel,
        "risk": LifestyleRiskModel,
        "chest_xray": ChestXrayModel,
    }

    IMAGE_MODELS = ["chest_xray"]
    TABULAR_MODELS = ["heart_disease", "risk"]

    @classmethod
    def get(cls, name: str):
        if name not in cls._CATALOG:
            raise ValueError(f"Unknown model '{name}'. Available: {list(cls._CATALOG)}")
        if name not in cls._instances:
            logger.info(f"Loading model '{name}' (first use)...")
            try:
                cls._instances[name] = cls._CATALOG[name]()
                cls._load_errors.pop(name, None)
            except ModelNotAvailableError as e:
                cls._load_errors[name] = str(e)
                raise
            except Exception as e:
                cls._load_errors[name] = f"Unexpected error loading model: {e}"
                raise ModelNotAvailableError(cls._load_errors[name]) from e
        return cls._instances[name]

    @classmethod
    def status(cls) -> dict[str, Any]:
        """Report per-model status without forcing a load: whether the weight file exists
        on disk, whether it's already loaded in memory, and the last load error if any."""
        report = {}
        for name, klass in cls._CATALOG.items():
            path = os.path.join(WEIGHTS_DIR, klass.WEIGHT_FILE)
            report[name] = {
                "weight_file": klass.WEIGHT_FILE,
                "weights_found_on_disk": os.path.exists(path),
                "loaded_in_memory": name in cls._instances,
                "last_error": cls._load_errors.get(name),
            }
        return report

    @classmethod
    def preload_all(cls) -> dict[str, str]:
        """Attempts to load every model eagerly (e.g. at API startup). Returns a dict of
        name -> 'ok' | error message, so the app can boot even if some models are missing."""
        results = {}
        for name in cls._CATALOG:
            try:
                cls.get(name)
                results[name] = "ok"
            except ModelNotAvailableError as e:
                results[name] = str(e)
        return results


def predict_image(model_name: str, image_bytes: bytes, with_gradcam: bool = True) -> PredictionResult:
    if model_name not in ModelRegistry.IMAGE_MODELS:
        raise ValueError(f"'{model_name}' is not an image model. Choose from {ModelRegistry.IMAGE_MODELS}")
    model = ModelRegistry.get(model_name)
    return model.predict(image_bytes, with_gradcam=with_gradcam)


def predict_tabular(model_name: str, features: dict[str, Any]) -> PredictionResult:
    if model_name not in ModelRegistry.TABULAR_MODELS:
        raise ValueError(f"'{model_name}' is not a tabular model. Choose from {ModelRegistry.TABULAR_MODELS}")
    model = ModelRegistry.get(model_name)
    return model.predict(features)

