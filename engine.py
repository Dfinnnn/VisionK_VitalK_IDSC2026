# =============================================================================
#  engine.py  —  Glaucoma AI Inference Engine
#  IDSC 2026  |  EfficientNet-B3 + Temperature Scaling + Grad-CAM + SHAP
#
#  THREE RULES:
#    1. No Streamlit imports here — pure logic only
#    2. No os.chdir() — uses paths relative to BASE_DIR
#    3. hygd_pipeline.py is never imported — functions re-implemented cleanly
#
#  USAGE (from app.py):
#    from engine import GlaucomaEngine
#    engine = GlaucomaEngine()                    # loads model once at startup
#    result = engine.predict(pil_image, threshold=0.3197)
# =============================================================================

import os
import json
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.special import expit as sigmoid
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ── Base directory (folder containing engine.py)
BASE_DIR = Path(__file__).resolve().parent

# ── Artifact paths
MODELS_DIR      = BASE_DIR / "models"
RESULTS_DIR     = BASE_DIR / "results"
CKPT_PATH       = MODELS_DIR / "glaucoma_efficientnet_b3.pth"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.json"
LGB_PATH        = MODELS_DIR / "lgb_shap_companion.pkl"
FEATURES_CSV    = RESULTS_DIR / "clinical_features.csv"


# =============================================================================
#  SECTION A — MODEL ARCHITECTURE
#  Exact replica of GlaucomaEfficientNet from hygd_pipeline.py.
#  Must match weights or load will fail.
# =============================================================================

class GlaucomaEfficientNet(nn.Module):
    """
    EfficientNet-B3 with custom binary classification head.
    Dropout(0.4) → Linear(1536→256) → BatchNorm → ReLU → Dropout(0.2) → Linear(256→1)
    Output: scalar logit → sigmoid → P(GON)
    """
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


# =============================================================================
#  SECTION B — PREPROCESSING
#  Must be IDENTICAL to training pipeline preprocessing.
#  Any deviation = distribution shift = wrong predictions.
# =============================================================================

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    CLAHE on LAB lightness channel + green channel unsharp mask.
    Adaptive tileGridSize based on image height (matches training exactly).
    """
    # ── CLAHE on L channel
    lab   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    tile  = max(1, img_rgb.shape[0] // 32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    img   = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ── Green channel unsharp mask (enhances RNFL structure)
    g       = img[:, :, 1].astype(np.float32)
    blurred = cv2.GaussianBlur(g, (0, 0), sigmaX=2)
    sharp   = np.clip(g + 1.5 * (g - blurred), 0, 255).astype(np.uint8)
    img[:, :, 1] = sharp
    return img


def preprocess_fundus(pil_image: Image.Image) -> Image.Image:
    """
    Converts PIL image → CLAHE-enhanced PIL image.
    Input:  PIL Image (any mode)
    Output: PIL Image (RGB, CLAHE + green sharpening applied)
    """
    img = np.array(pil_image.convert("RGB"))
    img = apply_clahe(img)
    return Image.fromarray(img)


def get_inference_transform(image_size: int = 300,
                             mean: list = None,
                             std:  list = None) -> transforms.Compose:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# =============================================================================
#  SECTION C — IMAGE QUALITY GATEKEEPER
#  Rejects low-quality images before prediction.
# =============================================================================

def check_image_quality(pil_image: Image.Image) -> dict:
    """
    Checks blur, brightness, and contrast.

    Returns:
        {
          "passed":      bool,
          "blur_score":  float,   # Laplacian variance — higher = sharper
          "brightness":  float,   # mean pixel value 0–255
          "contrast":    float,   # std of pixel values
          "warnings":    list[str],
          "reason":      str
        }
    """
    img  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast   = float(gray.std())

    warnings_list = []
    passed = True

    if blur_score < 50:
        warnings_list.append(
            f"Image appears blurry (sharpness score: {blur_score:.0f} — minimum 50). "
            "Please retake with the camera in focus."
        )
        passed = False
    elif blur_score < 100:
        warnings_list.append(
            f"Image sharpness is borderline ({blur_score:.0f}). "
            "Results may be less reliable."
        )

    if brightness < 20:
        warnings_list.append(
            f"Image is too dark (brightness: {brightness:.0f}). "
            "Check illumination and retake."
        )
        passed = False
    elif brightness > 230:
        warnings_list.append(
            f"Image is overexposed (brightness: {brightness:.0f}). "
            "Reduce illumination and retake."
        )
        passed = False

    if contrast < 15:
        warnings_list.append(
            f"Image has very low contrast ({contrast:.0f}). "
            "This may not be a valid fundus image."
        )
        passed = False

    reason = "OK" if passed else " | ".join(warnings_list)

    return {
        "passed":     passed,
        "blur_score": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "contrast":   round(contrast, 1),
        "warnings":   warnings_list,
        "reason":     reason,
    }


# =============================================================================
#  SECTION D — GRAD-CAM
#  Hooks on model.backbone.features[-1] — identical to training pipeline.
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Target layer: model.backbone.features[-1]
    """
    def __init__(self, model: GlaucomaEfficientNet, device: torch.device):
        self.model      = model
        self.device     = device
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.backbone.features[-1]

        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self, tensor: torch.Tensor) -> tuple:
        """
        Args:
            tensor: preprocessed image tensor (C, H, W) — no batch dim
        Returns:
            cam:  np.ndarray (H, W) normalised 0–1
            prob: float  raw sigmoid probability before calibration
        """
        self.model.eval()
        x   = tensor.unsqueeze(0).to(self.device)
        out = self.model(x)
        prob = float(torch.sigmoid(out).item())

        self.model.zero_grad()
        out.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam_np = cam.cpu().numpy()
        image_size = tensor.shape[-1]
        cam_np = cv2.resize(cam_np, (image_size, image_size))
        return cam_np, prob


def overlay_heatmap(original_rgb: np.ndarray,
                    cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Blends Grad-CAM heatmap onto original image.
    Returns RGB uint8 array same size as original.
    """
    h, w = original_rgb.shape[:2]
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (w, h))
    orig_r  = cv2.resize(original_rgb, (w, h))
    blended = (alpha * heatmap + (1 - alpha) * orig_r).astype(np.uint8)
    return blended


def validate_attention(cam: np.ndarray) -> dict:
    """
    Checks whether Grad-CAM focus is on the optic disc region (centre)
    rather than image borders (artifact learning).

    Returns:
        {
          "plausible":    bool,
          "center_pct":   float,
          "border_pct":   float,
          "note":         str,
          "warning":      str | None
        }
    """
    H, W       = cam.shape
    total_act  = cam.sum()

    if total_act == 0:
        return {
            "plausible":  False,
            "center_pct": 0.0,
            "border_pct": 0.0,
            "note":       "No activation detected",
            "warning":    "Reliability Warning: Grad-CAM produced no activation.",
        }

    centre_pct = float(
        cam[H // 3: 2 * H // 3, W // 3: 2 * W // 3].sum() / total_act * 100
    )
    bw = H // 8
    border_vals = np.concatenate([
        cam[:bw,  :].flatten(),
        cam[-bw:, :].flatten(),
        cam[:,  :bw].flatten(),
        cam[:, -bw:].flatten(),
    ])
    border_pct = float(border_vals.sum() / total_act * 100)
    plausible  = centre_pct > 25 and border_pct <= 40

    if border_pct > 40:
        note    = "Border attention — possible artifact learning"
        warning = (
            "Reliability Warning: The AI focused on image borders rather than "
            "the optic disc. This prediction may be unreliable. "
            "Consider retaking with better centration."
        )
    elif plausible:
        note    = "Central / disc region attention — clinically plausible"
        warning = None
    else:
        note    = "Peripheral attention — review this case"
        warning = (
            "Reliability Notice: AI attention is not centred on the optic disc. "
            "Interpret with caution."
        )

    return {
        "plausible":  plausible,
        "center_pct": round(centre_pct, 1),
        "border_pct": round(border_pct, 1),
        "note":       note,
        "warning":    warning,
    }


# =============================================================================
#  SECTION E — CLINICAL FEATURE EXTRACTION (for SHAP)
#  12 features matching exactly what LightGBM was trained on.
# =============================================================================

FEATURE_COLS = [
    "brightness_mean", "brightness_std",
    "red_channel_mean", "green_channel_mean", "blue_channel_mean",
    "rg_ratio", "centre_brightness", "periph_brightness",
    "centre_contrast", "disc_region_mean",
    "brightness_asymm", "laplacian_var",
]

FEATURE_LABELS = {
    "brightness_mean":   "Overall Brightness",
    "brightness_std":    "Brightness Variation",
    "red_channel_mean":  "Red Channel Intensity",
    "green_channel_mean":"Green Channel Intensity",
    "blue_channel_mean": "Blue Channel Intensity",
    "rg_ratio":          "Red:Green Ratio",
    "centre_brightness": "Central Brightness",
    "periph_brightness": "Peripheral Brightness",
    "centre_contrast":   "Central Contrast",
    "disc_region_mean":  "Disc Region Intensity",
    "brightness_asymm":  "Brightness Asymmetry",
    "laplacian_var":     "Image Sharpness",
}

SHAP_TRANSLATIONS = {
    "brightness_mean": (
        "Elevated overall retinal brightness may indicate disc pallor — a key GON marker.",
        "Normal overall brightness — no brightness-related GON indicator detected.",
    ),
    "brightness_std": (
        "High brightness variation suggests irregular retinal reflectance, consistent with RNFL loss.",
        "Uniform brightness distribution — no abnormal variation detected.",
    ),
    "red_channel_mean": (
        "Elevated red channel intensity may indicate disc haemorrhage or neovascularisation.",
        "Normal red channel — no haemorrhage indicator detected.",
    ),
    "green_channel_mean": (
        "Reduced green channel intensity suggests loss of RNFL reflectance.",
        "Normal green channel intensity — RNFL reflectance within expected range.",
    ),
    "blue_channel_mean": (
        "Abnormal blue channel intensity detected — possible disc colour abnormality.",
        "Normal blue channel — no colour abnormality detected.",
    ),
    "rg_ratio": (
        "Elevated red-to-green ratio may indicate disc haemorrhage or pallor — a GON risk factor.",
        "Normal red-to-green balance — no colour imbalance detected.",
    ),
    "centre_brightness": (
        "Increased central brightness may indicate cupping or disc pallor at the optic nerve head.",
        "Normal central brightness — optic disc appearance within expected range.",
    ),
    "periph_brightness": (
        "Reduced peripheral brightness suggests thinning of the neuroretinal rim (RNFL loss).",
        "Normal peripheral brightness — neuroretinal rim appears intact.",
    ),
    "centre_contrast": (
        "High central contrast suggests a sharp disc-cup boundary — consistent with enlarged cup.",
        "Soft disc-cup boundary — no abnormal cupping indicator detected.",
    ),
    "disc_region_mean": (
        "Elevated disc region intensity detected — possible disc pallor indicating optic nerve damage.",
        "Normal disc region intensity — no pallor detected.",
    ),
    "brightness_asymm": (
        "Significant brightness asymmetry between superior and inferior retina — consistent with focal RNFL loss.",
        "Symmetric brightness distribution — no significant RNFL asymmetry detected.",
    ),
    "laplacian_var": (
        "High image sharpness confirmed — structural detail is reliable for AI analysis.",
        "Lower image sharpness may reduce feature reliability. Interpret with caution.",
    ),
}


def extract_clinical_features(pil_image: Image.Image) -> dict:
    """
    Extracts 12 clinical proxy features from a fundus image.
    Must match feature extraction used during LightGBM training.

    Returns: dict with keys matching FEATURE_COLS
    """
    img  = np.array(pil_image.convert("RGB")).astype(np.float32)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    cy1, cy2 = H // 3, 2 * H // 3
    cx1, cx2 = W // 3, 2 * W // 3
    centre   = gray[cy1:cy2, cx1:cx2]

    dy1, dy2 = H // 4, 3 * H // 4
    dx1, dx2 = W // 4, 3 * W // 4
    disc     = gray[dy1:dy2, dx1:dx2]

    mask        = np.ones_like(gray, dtype=bool)
    mask[cy1:cy2, cx1:cx2] = False
    periph_vals = gray[mask]

    sup  = gray[:H // 2, :].mean()
    inf  = gray[H // 2:, :].mean()

    rg_ratio = float(r.mean() / (g.mean() + 1e-6))

    return {
        "brightness_mean":    float(gray.mean()),
        "brightness_std":     float(gray.std()),
        "red_channel_mean":   float(r.mean()),
        "green_channel_mean": float(g.mean()),
        "blue_channel_mean":  float(b.mean()),
        "rg_ratio":           round(rg_ratio, 4),
        "centre_brightness":  float(centre.mean()),
        "periph_brightness":  float(periph_vals.mean()),
        "centre_contrast":    float(centre.std()),
        "disc_region_mean":   float(disc.mean()),
        "brightness_asymm":   float(abs(sup - inf)),
        "laplacian_var":      float(cv2.Laplacian(
                                  gray.astype(np.uint8), cv2.CV_64F).var()),
    }


# =============================================================================
#  SECTION F — SHAP EXPLAINER
# =============================================================================

def compute_shap_values(lgb_model, features: dict) -> dict:
    """
    Computes per-feature SHAP values using the LightGBM companion model.

    Returns:
        {
          "shap_values":   dict[feature_name -> float],
          "base_value":    float,
          "top_features":  list of dicts sorted by |shap| descending, top 5
        }
    """
    import pandas as pd

    X = pd.DataFrame([features])[FEATURE_COLS]

    explainer   = shap.TreeExplainer(lgb_model)
    shap_vals   = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    shap_dict = {col: float(sv[i]) for i, col in enumerate(FEATURE_COLS)}

    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = []
    for feat, val in sorted_features[:5]:
        direction = "increases" if val > 0 else "decreases"
        label     = FEATURE_LABELS.get(feat, feat)
        idx       = 0 if val > 0 else 1
        sentence  = SHAP_TRANSLATIONS.get(feat, ("", ""))[idx]
        top_features.append({
            "feature":   feat,
            "label":     label,
            "shap_val":  round(val, 4),
            "direction": direction,
            "sentence":  sentence,
        })

    return {
        "shap_values":  shap_dict,
        "base_value":   float(explainer.expected_value[1]
                              if isinstance(explainer.expected_value, (list, np.ndarray))
                              else explainer.expected_value),
        "top_features": top_features,
    }


# =============================================================================
#  SECTION G — RISK SCORING
# =============================================================================

HIGH_RISK_THRESHOLD     = 0.60
MODERATE_RISK_THRESHOLD = 0.30

def compute_risk_score(cal_prob: float, threshold: float) -> dict:
    """
    Converts calibrated P(GON) to structured clinical output.

    Args:
        cal_prob:  calibrated probability 0–1
        threshold: decision threshold (from model_card or slider)

    Returns full risk dict including score, tier, color, recommendation,
    urgency, follow-up timeline, and confidence level.
    """
    score = round(float(cal_prob) * 100, 1)

    if score >= HIGH_RISK_THRESHOLD * 100:
        tier      = "High Risk"
        color     = "red"
        urgency   = "URGENT"
        action    = (
            "Immediate ophthalmology referral required. "
            "Arrange IOP measurement, Humphrey 24-2 visual field test, and OCT RNFL."
        )
        timeline  = "Within 2 weeks"
        evidence  = "Ting et al. 2017 (Nature Medicine) — high-risk deployment threshold"

    elif score >= MODERATE_RISK_THRESHOLD * 100:
        tier      = "Glaucoma Suspect"
        color     = "amber"
        urgency   = "MONITOR"
        action    = (
            "Ophthalmology review recommended. "
            "Arrange baseline OCT RNFL and IOP measurement."
        )
        timeline  = "Within 3 months"
        evidence  = "EGS Guidelines 2021 — glaucoma suspect management"

    else:
        tier      = "Low Risk"
        color     = "green"
        urgency   = "ROUTINE"
        action    = "Routine annual screening. No immediate action required."
        timeline  = "12 months"
        evidence  = "NICE guidelines — safe discharge NPV threshold"

    dist = abs(cal_prob - threshold)
    if dist >= 0.30:
        confidence       = "High"
        confidence_note  = "Prediction is well away from the decision boundary."
    elif dist >= 0.10:
        confidence       = "Moderate"
        confidence_note  = "Prediction has moderate certainty."
    else:
        confidence       = "Low — Boundary Case"
        confidence_note  = (
            "⚠ Prediction is close to the decision threshold. "
            "Clinical review strongly recommended."
        )

    gon_detected = cal_prob >= threshold

    return {
        "risk_score":        score,
        "risk_tier":         tier,
        "color":             color,
        "gon_detected":      gon_detected,
        "cal_prob":          round(float(cal_prob), 4),
        "threshold":         round(float(threshold), 4),
        "urgency":           urgency,
        "action":            action,
        "timeline":          timeline,
        "evidence":          evidence,
        "confidence":        confidence,
        "confidence_note":   confidence_note,
        "boundary_distance": round(dist, 4),
    }


# =============================================================================
#  SECTION H — MAIN ENGINE CLASS
#  Single object that app.py instantiates once via st.cache_resource.
# =============================================================================

class GlaucomaEngine:
    """
    Main inference engine. Load once, call predict() for each image.

    Attributes:
        model_card: dict  — full model metadata from model_card.json
        threshold:  float — default decision threshold (overridable per call)
        temperature:float — calibration temperature T
        device:     torch.device
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_card()
        self._load_model()
        self._load_lgb()
        self.gradcam = GradCAM(self.model, self.device)
        self.transform = get_inference_transform(
            image_size = self.model_card["image_size"],
            mean       = self.model_card["imagenet_mean"],
            std        = self.model_card["imagenet_std"],
        )
        print(f"  GlaucomaEngine ready — device: {self.device}")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_model_card(self):
        with open(MODEL_CARD_PATH) as f:
            self.model_card = json.load(f)
        self.threshold   = self.model_card["calibration"]["threshold"]
        self.temperature = self.model_card["calibration"]["temperature"]

    def _load_model(self):
        dropout = self.model_card.get("dropout_rate", 0.4)
        self.model = GlaucomaEfficientNet(dropout_rate=dropout).to(self.device)
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _load_lgb(self):
        self.lgb_model = joblib.load(LGB_PATH)

    # ── Calibration ──────────────────────────────────────────────────────────

    def _calibrate(self, logit: float) -> float:
        """Applies temperature scaling: sigmoid(logit / T)"""
        return float(sigmoid(np.array([logit]) / self.temperature)[0])

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self,
                pil_image: Image.Image,
                threshold: float = None,
                run_shap:  bool = True) -> dict:
        """
        Full inference pipeline for one fundus image.

        Args:
            pil_image: PIL Image — raw upload, any size/mode
            threshold: float | None — override default threshold (for slider)
            run_shap:  bool — set False to skip SHAP (faster, for quick preview)

        Returns:
            {
              "quality":           dict  — image quality check result
              "preprocessed_pil":  PIL Image  — CLAHE-enhanced image
              "risk":              dict  — risk score, tier, recommendation
              "gradcam":           dict  — cam array, overlay array, attention check
              "shap":              dict | None
              "logit":             float
              "raw_prob":          float
              "cal_prob":          float
            }
        """
        if threshold is None:
            threshold = self.threshold

        # Step 1: Image quality gatekeeper
        quality = check_image_quality(pil_image)

        # Step 2: Preprocessing (CLAHE + green sharpen)
        preprocessed_pil = preprocess_fundus(pil_image)

        # Step 3: Transform to tensor
        tensor = self.transform(preprocessed_pil)

        # Step 4: Model inference → logit
        with torch.no_grad():
            logit    = float(self.model(
                tensor.unsqueeze(0).to(self.device)
            ).item())
            raw_prob = float(torch.sigmoid(torch.tensor(logit)).item())

        # Step 5: Temperature scaling calibration
        cal_prob = self._calibrate(logit)

        # Step 6: Risk scoring
        risk = compute_risk_score(cal_prob, threshold)

        # Step 7: Grad-CAM
        cam_array, _ = self.gradcam.generate(tensor)
        orig_rgb     = np.array(pil_image.convert("RGB"))
        overlay      = overlay_heatmap(orig_rgb, cam_array)
        attention    = validate_attention(cam_array)

        # Step 8: SHAP
        shap_result = None
        if run_shap:
            features    = extract_clinical_features(preprocessed_pil)
            shap_result = compute_shap_values(self.lgb_model, features)

        return {
            "quality":           quality,
            "preprocessed_pil":  preprocessed_pil,
            "risk":              risk,
            "gradcam": {
                "cam":       cam_array,
                "overlay":   overlay,
                "attention": attention,
            },
            "shap":     shap_result,
            "logit":    round(logit, 4),
            "raw_prob": round(raw_prob, 4),
            "cal_prob": round(cal_prob, 4),
        }

    def get_model_card(self) -> dict:
        """Returns full model card metadata."""
        return self.model_card

    def get_default_threshold(self) -> float:
        return self.threshold

    def get_performance_metrics(self) -> dict:
        """Returns test set performance from model card."""
        return self.model_card.get("test_performance", {})