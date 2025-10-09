import os
import io
import re
import json
import base64
import hashlib
import math
from datetime import datetime
from typing import Optional, Dict, Any, List
from io import BytesIO

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image as PILImage

# Google Gemini
import google.generativeai as genai
from markdown import markdown

# ReportLab for PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import html

# Optional: skimage for advanced skeleton analysis
try:
    from skimage.morphology import skeletonize

    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Advanced AI Hair Analyzer",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# FIXED: Better API key handling
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyCR75y7nS8O-PrKqW7fmBDo2mtHBLd18CU"
)

# Initialize Gemini model
model = None
vision_model = None

if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        vision_model = genai.GenerativeModel("gemini-2.5-flash")
        st.sidebar.success("✅ Gemini API configured")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to configure Gemini: {str(e)[:100]}")
        model = None
        vision_model = None
else:
    st.sidebar.warning(
        "⚠️ Gemini API key not set. Please set GEMINI_API_KEY environment variable."
    )

USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cache for LLM responses
_llm_cache: Dict[str, Any] = {}
CACHE_TTL = 3600  # 1 hour

# Load face cascade
_FACE_CASCADE = None
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
except Exception:
    pass

# ==================== HELPER FUNCTIONS ====================


def _read_image(file_like_or_bytes) -> np.ndarray:
    """Read image from file or bytes"""
    data = (
        file_like_or_bytes.read()
        if hasattr(file_like_or_bytes, "read")
        else file_like_or_bytes
    )
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def profile_hash(profile: dict) -> str:
    """Generate hash for profile caching"""
    return hashlib.sha256(
        json.dumps(profile, sort_keys=True).encode("utf-8")
    ).hexdigest()


def cache_get(key: str):
    """Get cached result"""
    rec = _llm_cache.get(key)
    if not rec:
        return None
    ts, val = rec
    if (datetime.now().timestamp() - ts) > CACHE_TTL:
        del _llm_cache[key]
        return None
    return val


def cache_set(key: str, val: Any):
    """Set cached result"""
    _llm_cache[key] = (datetime.now().timestamp(), val)


# ==================== CV METRICS FUNCTIONS ====================


def estimate_hair_density(img: np.ndarray) -> dict:
    """Estimate hair density using edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    total_pixels = edges.size
    hair_pixels = np.count_nonzero(edges)
    density_score = hair_pixels / float(total_pixels)

    if density_score > 0.06:
        density_band = "high"
    elif density_score > 0.03:
        density_band = "medium"
    else:
        density_band = "low"

    scalp_exposed_ratio = 1 - density_score

    return {
        "hair_density_score": round(density_score, 4),
        "hair_density_band": density_band,
        "scalp_exposed_ratio": round(scalp_exposed_ratio, 4),
        "mask": edges,
    }


def estimate_bald_patch_area(img: np.ndarray) -> Dict[str, Any]:
    """Estimate bald patch area using HSV color detection"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    scalp_mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([179, 70, 255]))
    kernel = np.ones((5, 5), np.uint8)
    scalp_mask = cv2.morphologyEx(scalp_mask, cv2.MORPH_OPEN, kernel)
    scalp_mask = cv2.morphologyEx(scalp_mask, cv2.MORPH_CLOSE, kernel)

    scalp_pixels = int(np.count_nonzero(scalp_mask))
    total_pixels = int(scalp_mask.size)
    scalp_area_ratio = scalp_pixels / total_pixels if total_pixels > 0 else 0.0

    return {"scalp_exposed_ratio": round(scalp_area_ratio, 4), "mask": scalp_mask}


def estimate_scalp_redness(img: np.ndarray) -> dict:
    """Estimate scalp redness"""
    b, g, r = cv2.split(img)
    redness_index = max(0.0, float(np.mean(r) - np.mean(g)))
    redness_score = redness_index / 255.0

    if redness_score > 0.1:
        redness_band = "high"
    elif redness_score > 0.03:
        redness_band = "moderate"
    else:
        redness_band = "low"

    redness_mask = ((r > (g + 30)) & (r > (b + 30))).astype(np.uint8) * 255

    return {
        "redness_score": round(redness_score, 4),
        "redness_band": redness_band,
        "mask": redness_mask,
    }


# ==================== IMAGE QUALITY FUNCTIONS ====================


def _normalize_focus(var_lap: float, scale: float = 1200.0) -> float:
    """Normalize focus score"""
    return float(var_lap) / (float(var_lap) + scale)


def compute_image_quality(img: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive image quality metrics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())

    focus_score = _normalize_focus(var_lap, scale=1200.0)
    brightness_score = max(0.0, min(1.0, float(gray.mean()) / 255.0))
    contrast_score = max(0.0, min(1.0, float(gray.std()) / 64.0))

    face_score = 0.0
    try:
        if _FACE_CASCADE is not None:
            faces = _FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            face_score = 1.0 if len(faces) > 0 else 0.0
    except Exception:
        pass

    combined = (
        0.5 * focus_score
        + 0.3 * brightness_score
        + 0.15 * contrast_score
        + 0.05 * face_score
    )
    combined = max(0.0, min(1.0, combined))

    return {
        "focus_score": round(focus_score, 3),
        "brightness_score": round(brightness_score, 3),
        "contrast_score": round(contrast_score, 3),
        "face_detected": int(face_score),
        "image_quality_score": round(combined, 3),
    }


def compute_confidence_for_metrics(
    cv_metrics: Dict[str, Any], image_quality: Dict[str, float]
) -> Dict[str, Any]:
    """Compute confidence scores for CV metrics"""
    q = image_quality.get("image_quality_score", 0.0)
    confidences = {}

    density = float(cv_metrics.get("hair_density_score", 0.0))
    density_conf = q * (1.0 - abs(density - 0.06) / 0.12)
    density_conf = max(0.0, min(1.0, density_conf))
    confidences["hair_density_confidence_pct"] = int(round(density_conf * 100))

    scalp_ratio = float(cv_metrics.get("scalp_exposed_ratio", 0.0))
    scalp_conf = q * (0.5 + min(scalp_ratio, 0.5))
    scalp_conf = max(0.0, min(1.0, scalp_conf))
    confidences["scalp_exposed_confidence_pct"] = int(round(scalp_conf * 100))

    redness = float(cv_metrics.get("redness_score", 0.0))
    redness_conf = q * (0.5 + min(redness * 3.0, 0.5))
    redness_conf = max(0.0, min(1.0, redness_conf))
    confidences["redness_confidence_pct"] = int(round(redness_conf * 100))

    avg = int(
        round(
            (
                confidences["hair_density_confidence_pct"]
                + confidences["scalp_exposed_confidence_pct"]
                + confidences["redness_confidence_pct"]
            )
            / 3.0
        )
    )
    confidences["overall_confidence_pct"] = avg

    return confidences


# ==================== ADVANCED HAIR METRICS ====================


def simple_hair_mask_from_edges(edges):
    """Create hair mask from edges"""
    if edges is None:
        return None
    if edges.max() > 1:
        edges_bin = (edges > 0).astype(np.uint8)
    else:
        edges_bin = edges.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(edges_bin, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def estimate_diameter_distribution(mask_binary, pixel_size_mm: Optional[float] = None):
    """Estimate hair diameter distribution"""
    if mask_binary is None or mask_binary.sum() == 0:
        return {
            "diameters_um": [],
            "mean_um": None,
            "median_um": None,
            "std_um": None,
            "n_samples": 0,
        }

    dist = cv2.distanceTransform((mask_binary * 255).astype(np.uint8), cv2.DIST_L2, 5)

    if SKIMAGE_AVAILABLE:
        sk = skeletonize(mask_binary > 0)
        radii = dist[sk]
    else:
        thresh = max(1.0, dist.max() * 0.4)
        centers = dist >= thresh
        radii = dist[centers]
        if len(radii) == 0:
            radii = dist[dist > 0]

    diameters_px = radii * 2.0
    if pixel_size_mm:
        diameters_mm = diameters_px * float(pixel_size_mm)
        diameters_um = diameters_mm * 1000.0
    else:
        diameters_um = diameters_px

    diameters_um = np.array(diameters_um)
    if diameters_um.size == 0:
        return {
            "diameters_um": [],
            "mean_um": None,
            "median_um": None,
            "std_um": None,
            "n_samples": 0,
        }

    return {
        "diameters_um": diameters_um.tolist(),
        "mean_um": float(diameters_um.mean()),
        "median_um": float(np.median(diameters_um)),
        "std_um": float(diameters_um.std()),
        "n_samples": int(diameters_um.size),
    }


def classify_vellus_terminal(diameters_um_list, threshold_um: float = 40.0):
    """Classify hairs as vellus or terminal"""
    arr = np.array(diameters_um_list)
    if arr.size == 0:
        return {"vellus_count": 0, "terminal_count": 0, "vellus_terminal_ratio": None}
    v = int((arr < threshold_um).sum())
    t = int((arr >= threshold_um).sum())
    ratio = float(v) / float(v + t) if (v + t) > 0 else None
    return {"vellus_count": v, "terminal_count": t, "vellus_terminal_ratio": ratio}


def estimate_hair_count(mask_binary):
    """Estimate total hair count"""
    try:
        if SKIMAGE_AVAILABLE:
            sk = skeletonize(mask_binary > 0).astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            neighbors = cv2.filter2D(sk, -1, kernel) - sk
            endpoints = ((sk == 1) & (neighbors == 1)).sum()
            hair_count = int(max(1, endpoints))
        else:
            contours, _ = cv2.findContours(
                (mask_binary * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            hair_count = sum(1 for c in contours if cv2.arcLength(c, False) > 5)
    except Exception:
        hair_count = int((mask_binary > 0).sum() // 10)
    return hair_count


def compute_hairs_per_cm2(hair_count: int, img_shape, pixel_size_mm: Optional[float]):
    """Compute hairs per cm²"""
    h, w = img_shape[:2]
    if not pixel_size_mm or pixel_size_mm <= 0:
        return None
    area_mm2 = (h * pixel_size_mm) * (w * pixel_size_mm)
    area_cm2 = area_mm2 / 100.0
    if area_cm2 <= 0:
        return None
    return float(hair_count) / area_cm2


def local_density_heatmap(mask_binary, window_size_px=80, stride_px=40):
    """Generate local density heatmap"""
    h, w = mask_binary.shape
    heat = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, stride_px):
        for x in range(0, w, stride_px):
            y0 = max(0, y - window_size_px // 2)
            y1 = min(h, y + window_size_px // 2)
            x0 = max(0, x - window_size_px // 2)
            x1 = min(w, x + window_size_px // 2)
            region = mask_binary[y0:y1, x0:x1]
            count = (region > 0).sum()
            heat[y0:y1, x0:x1] += count
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
    heat_rgb = cv2.applyColorMap((heat_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat_rgb, heat_norm


# ==================== MASK OVERLAY ====================


def _ensure_mask(mask, img_shape):
    """Ensure mask is properly formatted"""
    h, w = img_shape[:2]
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    m = np.array(mask)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    if m.dtype != np.uint8:
        if m.max() <= 1.0:
            m = (m * 255).astype(np.uint8)
        else:
            m = m.astype(np.uint8)
    m = np.where(m > 0, 255, 0).astype(np.uint8)
    return m


def _color_mask(mask, color, img_shape):
    """Color a mask"""
    h, w = img_shape[:2]
    m = _ensure_mask(mask, img_shape)
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(color):
        colored[:, :, i] = (m // 255) * int(col)
    return colored


def create_mask_overlay_b64(
    img: np.ndarray, hair_mask, bald_mask, redness_mask, alpha=0.5
) -> str:
    """Create overlay image with colored masks"""
    hair_col = (0, 200, 0)
    bald_col = (200, 0, 0)
    red_col = (0, 0, 200)
    colored = np.zeros_like(img, dtype=np.uint8)
    colored = cv2.add(colored, _color_mask(hair_mask, hair_col, img.shape))
    colored = cv2.add(colored, _color_mask(bald_mask, bald_col, img.shape))
    colored = cv2.add(colored, _color_mask(redness_mask, red_col, img.shape))
    blended = cv2.addWeighted(img, 1.0, colored, alpha, 0)
    _, buf = cv2.imencode(".jpg", blended)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ==================== USER HISTORY ====================


def append_profile_history(
    profile_key: str,
    cv_metrics: Dict[str, Any],
    image_quality: Dict[str, Any],
    advanced_metrics: Optional[Dict[str, Any]] = None,
):
    """Append scan to user history"""
    fname = os.path.join(USER_DATA_DIR, f"{profile_key}.json")
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cv_metrics": cv_metrics,
        "image_quality": image_quality,
        "advanced_metrics": advanced_metrics or {},
    }
    try:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        data = data[-100:]  # Keep last 100 scans
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")


def load_profile_history(profile_key: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Load user history"""
    fname = os.path.join(USER_DATA_DIR, f"{profile_key}.json")
    if not os.path.exists(fname):
        return []
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[-limit:]
    except Exception:
        return []


# ==================== AI ANALYSIS ====================


def analyze_hair_image_gemini(image_path):
    """Analyze image with Gemini Vision - IMPROVED VERSION"""
    if vision_model is None:
        return {
            "hair_type": "API Not Configured",
            "scalp_condition": "API Not Configured",
            "issues": [
                "Gemini API not configured. Please set GEMINI_API_KEY environment variable."
            ],
        }

    prompt = """You are an expert dermatologist AI specializing in hair and scalp analysis.

Analyze this scalp/hair photo carefully and provide a structured analysis.

CRITICAL: Return ONLY a valid JSON object with NO markdown formatting, NO code blocks, NO extra text.

Required JSON structure:
{
  "hair_type": "one of: Straight, Wavy, Curly, Coily, Mixed",
  "scalp_condition": "one of: Healthy, Oily, Dry, Flaky, Inflamed, Combination",
  "issues": ["list specific visible issues"]
}

Analyze for:
- Hair texture and pattern (straight/wavy/curly/coily)
- Scalp appearance (color, texture, dryness/oiliness)
- Visible issues: hair loss, thinning, receding hairline, bald patches, dandruff, dryness, oiliness, redness, inflammation, breakage

Be specific and professional. Return ONLY the JSON object."""

    try:
        # Read image
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        # Create image part for Gemini
        image_part = {"mime_type": "image/jpeg", "data": image_data}

        # Configure generation with safety settings
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=800,
            candidate_count=1,
        )

        # Safety settings to avoid blocks
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Generate content
        response = vision_model.generate_content(
            [prompt, image_part],
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # Check if response was blocked
        if not response.candidates:
            st.warning("⚠️ AI response was blocked. Trying alternative analysis...")
            return {
                "hair_type": "Unable to determine",
                "scalp_condition": "Unable to determine",
                "issues": [
                    "AI analysis was blocked. Please try with a different image."
                ],
            }

        # Get the text response
        if hasattr(response, "text"):
            ai_text = response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                ai_text = "".join(
                    [
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text")
                    ]
                )
            else:
                raise ValueError("Could not extract text from response")
        else:
            raise ValueError("No valid response received from AI")

        # Clean the response
        ai_text = ai_text.strip()

        # Remove markdown code blocks if present
        ai_text = re.sub(r"^```(?:json)?\s*", "", ai_text)
        ai_text = re.sub(r"\s*```$", "", ai_text)
        ai_text = ai_text.strip()

        # Try to find JSON object in the text
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", ai_text, re.DOTALL)

        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)

            # Validate required fields
            required_fields = ["hair_type", "scalp_condition", "issues"]
            if all(k in result for k in required_fields):
                # Ensure issues is a list
                if not isinstance(result["issues"], list):
                    result["issues"] = [str(result["issues"])]
                return result
            else:
                raise ValueError(f"Missing required fields. Got: {list(result.keys())}")
        else:
            # If no JSON found, try to parse as plain JSON
            result = json.loads(ai_text)
            if all(k in result for k in ["hair_type", "scalp_condition", "issues"]):
                if not isinstance(result["issues"], list):
                    result["issues"] = [str(result["issues"])]
                return result
            else:
                raise ValueError("Invalid JSON structure")

    except json.JSONDecodeError as je:
        st.error(f"JSON parsing error: {str(je)}")
        st.code(f"AI Response:\n{ai_text[:500]}")
        return {
            "hair_type": "Parse Error",
            "scalp_condition": "Parse Error",
            "issues": [f"Could not parse AI response: {str(je)[:200]}"],
        }
    except Exception as e:
        error_msg = str(e)
        st.error(f"AI Vision Analysis Error: {error_msg}")

        # Return fallback analysis
        return {
            "hair_type": "Analysis Failed",
            "scalp_condition": "Analysis Failed",
            "issues": [
                f"AI analysis encountered an error. Please try again with a clearer, well-lit image of your scalp."
            ],
        }


def analyze_hair_results_gemini(image_path, analysis_results, cv_metrics):
    """Generate comprehensive analysis with Gemini - IMPROVED VERSION"""
    if model is None:
        return {
            "analysis_summary": "API Configuration Error",
            "ai_suggestions": "⚠️ Gemini API not properly configured. Please set GEMINI_API_KEY environment variable to get AI-powered recommendations.",
        }

    try:
        # Extract values safely
        hair_type = analysis_results.get("hair_type", "Not Available")
        scalp_condition = analysis_results.get("scalp_condition", "Not Available")
        issues_list = analysis_results.get("issues", ["None detected"])

        # Format issues
        if isinstance(issues_list, list):
            issues_str = ", ".join(issues_list) if issues_list else "None detected"
        else:
            issues_str = str(issues_list)

        # Build comprehensive prompt
        prompt = f"""You are an expert dermatologist AI providing professional hair and scalp analysis.

**PATIENT DATA:**
- Hair Type (AI Detected): {hair_type}
- Scalp Condition (AI Detected): {scalp_condition}
- Visible Issues: {issues_str}
- Age: {analysis_results.get("age", "Not Provided")}
- Sex: {analysis_results.get("sex", "Not Provided")}
- Family History of Hair Loss: {"Yes" if analysis_results.get("family_history") else "No"}
- Stress Level (1-10): {analysis_results.get("stress", "Not Provided")}/10
- Diet Quality (1-10): {analysis_results.get("diet_quality", "Not Provided")}/10
- Sleep Hours: {analysis_results.get("sleep_hours", "Not Provided")} hours/night
- Treatment Preference: {analysis_results.get("regimen_strength", "Not Provided")}

**COMPUTER VISION METRICS:**
- Hair Density Score: {cv_metrics.get("hair_density_score", "N/A")} ({cv_metrics.get("hair_density_band", "N/A")})
- Scalp Exposed: {cv_metrics.get("scalp_exposed_ratio", 0)*100:.1f}%
- Redness Score: {cv_metrics.get("redness_score", "N/A")} ({cv_metrics.get("redness_band", "N/A")})

**INSTRUCTIONS:**
Provide a comprehensive professional analysis in markdown format with these exact sections:

## 📋 Summary
- Overall scalp and hair health assessment
- Primary concerns identified
- Severity classification (Mild/Moderate/Severe)

## 🔍 Detailed Analysis
### Hair and Scalp Findings
- Analysis of detected hair type and scalp condition
- Interpretation of visible issues
- Computer vision insights correlation

### Contributing Factors
- Lifestyle factors impact (stress, diet, sleep)
- Family history implications
- Environmental and behavioral factors

## 💊 Treatment Recommendations
### Immediate Actions (Next 2 Weeks)
- 3-4 specific actionable steps

### Hair Care Products
- Recommended shampoo types and frequency
- Conditioner recommendations
- Scalp treatments (if needed)
- Leave-in products (if applicable)

### Supplements & Nutrition
- Recommended supplements (if needed)
- Dietary improvements
- Hydration guidelines

### Lifestyle Modifications
- Stress management techniques
- Sleep optimization
- Exercise recommendations

## 📅 Daily Hair Care Regimen
### Morning Routine
- Specific morning care steps

### Evening Routine
- Evening care steps

### Weekly Treatments
- Deep conditioning protocol
- Scalp massage techniques
- Exfoliation (if needed)

## ⚕️ Medical Guidance
### When to Consult a Doctor
- Clear indicators for professional consultation
- Type of specialist recommended
- Urgency level

### Treatment Options Discussion
- Non-surgical treatments overview
- Prescription treatments (if applicable)
- Surgical options (if severe hair loss detected)

## 🎯 Progress Tracking
### What to Monitor
- Specific metrics to track monthly
- Photography guidelines for comparison
- Symptom diary suggestions

### Expected Timeline
- Short-term improvements (2-4 weeks)
- Medium-term results (2-3 months)
- Long-term outcomes (6-12 months)

### When to Reassess
- Signs indicating need for treatment adjustment
- Follow-up schedule recommendations

## 🚫 What to Avoid
- Products/ingredients to avoid
- Harmful styling practices
- Common mistakes

**IMPORTANT:**
- Be professional, empathetic, and encouraging
- Provide specific, actionable advice
- Avoid specific brand names (use generic categories)
- Emphasize prevention and maintenance
- Set realistic expectations
- Always encourage professional consultation for severe cases"""

        # Read image
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        image_part = {"mime_type": "image/jpeg", "data": image_data}

        # Generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=4000,
            candidate_count=1,
        )

        # Safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Generate comprehensive analysis
        response = model.generate_content(
            [prompt, image_part],
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # Extract text from response
        if hasattr(response, "text") and response.text:
            ai_text = response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                ai_text = "".join(
                    [
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text")
                    ]
                )
            else:
                ai_text = None
        else:
            ai_text = None

        if ai_text and len(ai_text) > 100:
            return {
                "analysis_summary": "AI-Powered Professional Hair & Scalp Analysis",
                "ai_suggestions": ai_text,
            }
        else:
            return {
                "analysis_summary": "Analysis Incomplete",
                "ai_suggestions": "⚠️ AI did not generate a complete response. Please try again with a clearer, well-lit image showing your scalp clearly.",
            }

    except Exception as e:
        error_msg = str(e)
        st.error(f"AI Analysis Error: {error_msg[:300]}")
        return {
            "analysis_summary": "Analysis Error",
            "ai_suggestions": f"""⚠️ **AI analysis encountered an error**

The system was unable to complete the comprehensive analysis. This could be due to:
- API connectivity issues
- Image quality concerns
- Rate limiting

**Recommended Actions:**
1. Ensure your image is clear and well-lit
2. Check your internet connection
3. Wait a moment and try again
4. If the issue persists, try uploading a different image

**Error Details:** {error_msg[:500]}""",
        }


# ==================== PDF GENERATION ====================


def generate_pdf_report(ai_suggestions, scalp_image_path=None):
    """Generate PDF report - FIXED VERSION"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("AI Hair & Scalp Analysis Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(
            Paragraph(
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 20))

        # Add image if provided - FIXED
        if scalp_image_path and os.path.exists(scalp_image_path):
            try:
                # Open and convert image properly
                with PILImage.open(scalp_image_path) as img:
                    # Convert to RGB if needed
                    if img.mode in ("RGBA", "LA", "P"):
                        img = img.convert("RGB")

                    # Resize for PDF
                    img.thumbnail((400, 300), PILImage.Resampling.LANCZOS)

                    # Save to a BytesIO buffer instead of temp file
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="JPEG", quality=85)
                    img_buffer.seek(0)

                    # Add to PDF
                    story.append(
                        ReportLabImage(
                            img_buffer, width=img.size[0], height=img.size[1]
                        )
                    )
                    story.append(Spacer(1, 20))
            except Exception as img_err:
                st.warning(f"Could not add image to PDF: {str(img_err)}")
                story.append(
                    Paragraph("(Scalp image could not be included)", styles["Italic"])
                )
                story.append(Spacer(1, 12))

        # Add analysis content
        if ai_suggestions and ai_suggestions.strip():
            # Process markdown content for PDF
            lines = ai_suggestions.replace("\r\n", "\n").replace("\r", "\n").split("\n")

            for line in lines:
                line = line.strip()

                if not line:
                    story.append(Spacer(1, 6))
                    continue

                # Handle headers
                if line.startswith("###"):
                    text = line.replace("###", "").strip()
                    story.append(Paragraph(text, styles["Heading3"]))
                    story.append(Spacer(1, 6))
                elif line.startswith("##"):
                    text = line.replace("##", "").strip()
                    story.append(Paragraph(text, styles["Heading2"]))
                    story.append(Spacer(1, 8))
                elif line.startswith("#"):
                    text = line.replace("#", "").strip()
                    story.append(Paragraph(text, styles["Heading1"]))
                    story.append(Spacer(1, 10))
                # Handle bullets
                elif line.startswith("- ") or line.startswith("* "):
                    text = line[2:].strip()
                    safe_text = html.escape(text)
                    story.append(Paragraph(f"• {safe_text}", styles["Normal"]))
                    story.append(Spacer(1, 4))
                # Handle numbered lists
                elif re.match(r"^\d+\.", line):
                    safe_text = html.escape(line)
                    story.append(Paragraph(safe_text, styles["Normal"]))
                    story.append(Spacer(1, 4))
                # Regular paragraphs
                else:
                    safe_text = html.escape(line)
                    # Handle bold
                    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)
                    # Handle italic
                    safe_text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", safe_text)
                    story.append(Paragraph(safe_text, styles["Normal"]))
                    story.append(Spacer(1, 6))
        else:
            story.append(
                Paragraph(
                    "No analysis content available. Please ensure AI analysis completed successfully.",
                    styles["Normal"],
                )
            )

        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("_" * 80, styles["Normal"]))
        story.append(Spacer(1, 10))
        story.append(
            Paragraph(
                "This report is generated by AI and should not replace professional medical advice.",
                styles["Italic"],
            )
        )
        story.append(
            Paragraph(
                "Please consult a dermatologist or trichologist for clinical diagnosis.",
                styles["Italic"],
            )
        )

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return io.BytesIO()


# ==================== UTILITY FUNCTIONS ====================


def retake_guidance_text(quality: dict) -> str:
    """Generate guidance for image retake"""
    tips = []
    if quality.get("focus_score", 0) < 0.45:
        tips.append("Hold your camera steady and ensure the scalp area is in focus.")
    if quality.get("brightness_score", 0) < 0.35:
        tips.append("Increase ambient light or move to a brighter area.")
    if quality.get("brightness_score", 0) > 0.9:
        tips.append("Avoid direct harsh sunlight; diffuse the light.")
    if quality.get("contrast_score", 0) < 0.3:
        tips.append("Use a plain background and even lighting.")
    if quality.get("face_detected", 0) == 0:
        tips.append("Make sure the scalp/head is clearly visible.")
    if not tips:
        tips.append("Image looks good. Proceed with analysis.")
    return " • ".join(tips)


def safe_pct_change(baseline_val, latest_val):
    """Calculate percentage change safely"""
    try:
        if baseline_val is None or (
            isinstance(baseline_val, (int, float))
            and math.isclose(float(baseline_val), 0.0, abs_tol=1e-12)
        ):
            return None
        return (float(latest_val) - float(baseline_val)) / float(baseline_val) * 100.0
    except Exception:
        return None


# ==================== STREAMLIT UI ====================

st.title("🧬 Advanced AI Hair Health Analyzer")
st.markdown(
    "**Professional hair and scalp analysis with user tracking, advanced metrics, and AI-powered recommendations**"
)

# Custom CSS
st.markdown(
    """
<style>
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📋 User Profile & Tracking")

    # User ID for tracking
    user_id = st.text_input(
        "👤 Enter Tracking ID (required)",
        value="",
        help="Use the same ID for all your scans to track progress over time",
    )
    profile_name = st.text_input("📝 Profile Name (optional)", value="default")

    st.markdown("---")

    # Image upload
    st.header("📸 Upload Scalp Photo")
    uploaded_photo = st.file_uploader("Upload clear photo", type=["jpg", "jpeg", "png"])

    st.markdown("---")

    # Personal Information
    st.header("👤 Personal Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female", "Other", "Prefer not to say"])
    family_history = st.radio("Family history of hair loss?", ["Yes", "No"], index=1)

    # Hair & Scalp Details
    st.subheader("💇 Hair & Scalp")
    hair_type = st.selectbox("Hair Type", ["", "Straight", "Wavy", "Curly", "Coily"])
    scalp_condition = st.selectbox(
        "Scalp Condition", ["", "Normal", "Oily", "Dry", "Itchy", "Dandruff"]
    )
    issues = st.multiselect(
        "Primary Concerns",
        [
            "Hair loss",
            "Thinning",
            "Breakage",
            "Dandruff",
            "Itchiness",
            "Oily scalp",
            "Dryness",
        ],
    )

    # Lifestyle Factors
    st.subheader("🏃 Lifestyle")
    stress = st.slider("Stress Level (1=Low, 10=High)", 1, 10, 5)
    diet_quality = st.slider("Diet Quality (1=Poor, 10=Excellent)", 1, 10, 6)
    sleep_hours = st.slider("Average Sleep (hours/night)", 0.0, 16.0, 7.5, 0.5)

    # Treatment Preferences
    st.subheader("💊 Treatment Preferences")
    regimen_strength = st.selectbox(
        "Regimen Intensity", ["Minimalist", "Standard", "Intensive"]
    )

    # Advanced Settings
    st.subheader("⚙️ Advanced Settings")
    pixel_size_mm = st.number_input(
        "Pixel size (mm) - optional calibration",
        min_value=0.0,
        value=0.0,
        step=0.001,
        help="Leave at 0 for uncalibrated analysis. For dermatoscope images, enter actual pixel size.",
    )
    vellus_threshold_um = st.number_input(
        "Vellus/Terminal threshold (μm)", min_value=1.0, value=40.0, step=1.0
    )

    st.markdown("---")

    # Privacy Controls
    st.subheader("🔒 Privacy & Data")
    local_only = st.checkbox("Process locally only (skip AI)", value=False)
    consent_research = st.checkbox("Opt-in to anonymized research", value=False)

    # History Management
    st.markdown("---")
    st.subheader("📊 History Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Delete History"):
            if not user_id:
                st.error("Enter ID first")
            else:
                key = profile_hash({"user_id": user_id, "profile_name": profile_name})
                fpath = os.path.join(USER_DATA_DIR, f"{key}.json")
                if os.path.exists(fpath):
                    os.remove(fpath)
                    st.success("History deleted!")
                else:
                    st.info("No history found")

    with col2:
        if st.button("📥 Export CSV"):
            if not user_id:
                st.error("Enter ID first")
            else:
                key = profile_hash({"user_id": user_id, "profile_name": profile_name})
                history = load_profile_history(key)
                if history:
                    rows = []
                    for e in history:
                        row = {"timestamp": e.get("timestamp")}
                        row.update(
                            {
                                f"cv_{k}": v
                                for k, v in e.get("cv_metrics", {}).items()
                                if k != "mask"
                            }
                        )
                        row.update(
                            {
                                f"iq_{k}": v
                                for k, v in e.get("image_quality", {}).items()
                            }
                        )
                        am = e.get("advanced_metrics", {})
                        if am:
                            row["mean_diameter_um"] = am.get("diameter", {}).get(
                                "mean_um"
                            )
                            row["hairs_per_cm2"] = am.get("hairs_per_cm2")
                        rows.append(row)
                    df = pd.DataFrame(rows)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download",
                        data=csv,
                        file_name=f"history_{user_id}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No history to export")

    st.markdown("---")
    analyze_button = st.button(
        "✨ Analyze Hair Health", type="primary", use_container_width=True
    )

# ==================== MAIN CONTENT ====================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("📷 Your Image")

    if uploaded_photo:
        try:
            photo_bytes = uploaded_photo.getvalue()
            img = _read_image(photo_bytes)
            image_quality = compute_image_quality(img)

            st.image(photo_bytes, caption="Uploaded Image", use_container_width=True)

            # Image Quality Display
            st.markdown("### 📊 Image Quality Assessment")
            quality_score = image_quality["image_quality_score"]

            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            col_q1.metric("Overall", f"{int(quality_score*100)}%")
            col_q2.metric("Focus", f"{image_quality['focus_score']:.2f}")
            col_q3.metric("Brightness", f"{image_quality['brightness_score']:.2f}")
            col_q4.metric("Contrast", f"{image_quality['contrast_score']:.2f}")

            if quality_score < 0.5:
                guidance = retake_guidance_text(image_quality)
                st.warning(f"⚠️ **Image quality is low.** {guidance}")
            else:
                st.success("✅ Image quality is good!")

        except Exception as e:
            st.error(f"Could not read image: {e}")
    else:
        st.info("👆 Upload a scalp photo in the sidebar to begin analysis")
        st.image(
            "https://img.freepik.com/premium-photo/indian-model-showing-great-hair-beauty-ad-white-background_878783-10574.jpg",
            caption="Example - Clear, well-lit photo needed",
            use_container_width=True,
        )

with col_right:
    st.header("🔬 Analysis Results")

    if analyze_button:
        if not user_id:
            st.warning(
                "⚠️ Please enter a **Tracking ID** in the sidebar to save your scan history."
            )
        elif not uploaded_photo:
            st.warning("⚠️ Please upload a photo before analyzing.")
        else:
            with st.spinner("🧪 Analyzing hair health... Please wait..."):
                try:
                    # Read image
                    photo_bytes = uploaded_photo.getvalue()
                    img = _read_image(photo_bytes)

                    # Save image temporarily
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{user_id}.jpg")
                    cv2.imwrite(image_path, img)

                    # CV Metrics
                    st.info("📊 Computing computer vision metrics...")
                    density = estimate_hair_density(img)
                    bald = estimate_bald_patch_area(img)
                    redness = estimate_scalp_redness(img)

                    cv_metrics = {
                        **{k: v for k, v in density.items() if k != "mask"},
                        **{k: v for k, v in bald.items() if k != "mask"},
                        **{k: v for k, v in redness.items() if k != "mask"},
                    }

                    # Image Quality
                    image_quality = compute_image_quality(img)
                    confidences = compute_confidence_for_metrics(
                        cv_metrics, image_quality
                    )

                    # Advanced Metrics
                    st.info("🔬 Calculating advanced hair metrics...")
                    hair_mask = simple_hair_mask_from_edges(density.get("mask"))
                    pixel_cal = pixel_size_mm if pixel_size_mm > 0 else None
                    diam_res = estimate_diameter_distribution(
                        hair_mask, pixel_size_mm=pixel_cal
                    )
                    vellus_res = classify_vellus_terminal(
                        diam_res.get("diameters_um", []),
                        threshold_um=vellus_threshold_um,
                    )
                    hair_count = estimate_hair_count(hair_mask)
                    hairs_per_cm2_val = compute_hairs_per_cm2(
                        hair_count, img.shape, pixel_cal
                    )
                    heat_rgb, heat_norm = local_density_heatmap(hair_mask)

                    advanced_metrics = {
                        "diameter": {
                            "mean_um": diam_res.get("mean_um"),
                            "median_um": diam_res.get("median_um"),
                            "std_um": diam_res.get("std_um"),
                            "n_samples": diam_res.get("n_samples"),
                        },
                        "vellus_terminal": vellus_res,
                        "hair_count": hair_count,
                        "hairs_per_cm2": hairs_per_cm2_val,
                    }

                    # Initialize session state
                    if "analysis_results" not in st.session_state:
                        st.session_state.analysis_results = {}

                    st.session_state.analysis_results = {
                        "cv_metrics": cv_metrics,
                        "image_quality": image_quality,
                        "confidences": confidences,
                        "advanced_metrics": advanced_metrics,
                        "image_path": image_path,
                        "density_mask": density.get("mask"),
                        "bald_mask": bald.get("mask"),
                        "redness_mask": redness.get("mask"),
                        "heat_rgb": heat_rgb,
                        "diam_res": diam_res,
                    }

                    # AI Analysis
                    if not local_only and (
                        model is not None or vision_model is not None
                    ):
                        st.info("🤖 Running AI vision analysis...")
                        ai_analysis = analyze_hair_image_gemini(image_path)

                        st.info("🧠 Generating comprehensive AI recommendations...")
                        full_result = analyze_hair_results_gemini(
                            image_path,
                            {
                                **ai_analysis,
                                "age": age,
                                "sex": sex,
                                "family_history": family_history == "Yes",
                                "stress": stress,
                                "diet_quality": diet_quality,
                                "sleep_hours": sleep_hours,
                                "regimen_strength": regimen_strength,
                            },
                            cv_metrics,
                        )

                        st.session_state.analysis_results["ai_result"] = full_result
                        st.session_state.analysis_results["ai_analysis"] = ai_analysis
                    else:
                        st.session_state.analysis_results["ai_result"] = {
                            "analysis_summary": "Local-only analysis (AI skipped)",
                            "ai_suggestions": "AI analysis was skipped. Uncheck 'Process locally only' in sidebar and ensure GEMINI_API_KEY is set to get AI recommendations.",
                        }
                        st.session_state.analysis_results["ai_analysis"] = {
                            "hair_type": "Not Analyzed",
                            "scalp_condition": "Not Analyzed",
                            "issues": ["AI analysis was skipped"],
                        }

                    # Save history
                    history_key = profile_hash(
                        {"user_id": user_id, "profile_name": profile_name}
                    )
                    append_profile_history(
                        history_key, cv_metrics, image_quality, advanced_metrics
                    )

                    st.success("✅ Analysis complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    import traceback

                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# ==================== DISPLAY RESULTS ====================
if "analysis_results" in st.session_state and st.session_state.analysis_results:
    results = st.session_state.analysis_results

    st.markdown("---")

    # Tabs for organized results
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Metrics", "🔬 Advanced", "🤖 AI Analysis", "📈 Trends", "📄 Report"]
    )

    with tab1:
        st.subheader("Computer Vision Metrics")

        col_m1, col_m2, col_m3 = st.columns(3)
        cv_m = results["cv_metrics"]

        with col_m1:
            st.metric(
                "Hair Density",
                cv_m.get("hair_density_band", "N/A").upper(),
                f"Score: {cv_m.get('hair_density_score', 0):.4f}",
            )
        with col_m2:
            st.metric(
                "Scalp Exposure", f"{cv_m.get('scalp_exposed_ratio', 0)*100:.1f}%"
            )
        with col_m3:
            st.metric(
                "Redness",
                cv_m.get("redness_band", "N/A").upper(),
                f"Score: {cv_m.get('redness_score', 0):.4f}",
            )

        st.markdown("### 🎯 Confidence Scores")
        conf = results["confidences"]
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Overall", f"{conf.get('overall_confidence_pct', 0)}%")
        col_c2.metric("Density", f"{conf.get('hair_density_confidence_pct', 0)}%")
        col_c3.metric("Exposure", f"{conf.get('scalp_exposed_confidence_pct', 0)}%")
        col_c4.metric("Redness", f"{conf.get('redness_confidence_pct', 0)}%")

        if conf.get("overall_confidence_pct", 0) < 60:
            st.info(
                "💡 Confidence is moderate. Consider retaking photo with better lighting for more accurate results."
            )

        # Overlay visualization
        st.markdown("### 🖼️ Visual Analysis")
        try:
            overlay_b64 = create_mask_overlay_b64(
                cv2.imread(results["image_path"]),
                results["density_mask"],
                results["bald_mask"],
                results["redness_mask"],
            )
            overlay_bytes = base64.b64decode(overlay_b64)
            st.image(
                overlay_bytes,
                caption="Overlay: Green=Hair, Red=Bald areas, Blue=Redness",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not generate overlay image: {str(e)}")

    with tab2:
        st.subheader("Advanced Hair Metrics")

        adv = results["advanced_metrics"]

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Hair Count", adv.get("hair_count", "N/A"))
        with col_a2:
            density_val = adv.get("hairs_per_cm2")
            st.metric("Hairs/cm²", f"{density_val:.1f}" if density_val else "N/A")
        with col_a3:
            mean_diam = adv.get("diameter", {}).get("mean_um")
            st.metric("Avg Diameter", f"{mean_diam:.1f} μm" if mean_diam else "N/A")

        # Vellus/Terminal
        st.markdown("### 🔬 Hair Classification")
        vt = adv.get("vellus_terminal", {})
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("Vellus Hairs", vt.get("vellus_count", 0))
        col_v2.metric("Terminal Hairs", vt.get("terminal_count", 0))
        ratio = vt.get("vellus_terminal_ratio")
        col_v3.metric("V/T Ratio", f"{ratio:.2f}" if ratio else "N/A")

        # Diameter distribution histogram
        st.markdown("### 📊 Diameter Distribution")
        diameters = results["diam_res"].get("diameters_um", [])
        if diameters and len(diameters) > 5:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(diameters, bins=25, color="steelblue", edgecolor="black")
            ax.set_xlabel("Diameter (μm)")
            ax.set_ylabel("Count")
            ax.set_title("Hair Diameter Distribution")
            ax.axvline(
                x=40, color="red", linestyle="--", label="Vellus/Terminal threshold"
            )
            ax.legend()
            st.pyplot(fig)
        else:
            st.info(
                "Not enough diameter samples. Improve image quality or use calibrated dermatoscope."
            )

        # Density heatmap
        st.markdown("### 🔥 Local Density Heatmap")
        try:
            _, buf = cv2.imencode(".jpg", results["heat_rgb"])
            st.image(
                buf.tobytes(),
                caption="Heatmap: Red=High density, Blue=Low density",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not display heatmap: {str(e)}")

    with tab3:
        st.subheader("🤖 AI-Powered Analysis")

        if "ai_result" in results:
            # Show AI Detection Results first
            if "ai_analysis" in results:
                st.markdown("### 👁️ AI Vision Detection")
                ai_det = results["ai_analysis"]

                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    st.info(f"**Hair Type:** {ai_det.get('hair_type', 'N/A')}")
                with col_ai2:
                    st.info(
                        f"**Scalp Condition:** {ai_det.get('scalp_condition', 'N/A')}"
                    )

                issues_detected = ai_det.get("issues", [])
                if issues_detected:
                    st.warning(f"**Detected Issues:** {', '.join(issues_detected)}")

                st.markdown("---")

            # Show comprehensive analysis
            ai_res = results["ai_result"]
            st.markdown(f"### {ai_res.get('analysis_summary', 'Analysis')}")

            # Render markdown suggestions
            suggestions_md = ai_res.get("ai_suggestions", "No suggestions available")
            st.markdown(suggestions_md)
        else:
            st.info(
                "AI analysis not available. Uncheck 'Process locally only' in sidebar and re-analyze."
            )

    with tab4:
        st.subheader("📈 Historical Trends")

        if user_id:
            history_key = profile_hash(
                {"user_id": user_id, "profile_name": profile_name}
            )
            history = load_profile_history(history_key)

            if len(history) > 1:
                rows = []
                for e in history:
                    ts = e.get("timestamp")
                    cm = e.get("cv_metrics", {})
                    am = e.get("advanced_metrics", {})
                    rows.append(
                        {
                            "timestamp": ts,
                            "hair_density_score": cm.get("hair_density_score"),
                            "scalp_exposed_ratio": cm.get("scalp_exposed_ratio"),
                            "redness_score": cm.get("redness_score"),
                            "mean_diameter_um": am.get("diameter", {}).get("mean_um"),
                            "hairs_per_cm2": am.get("hairs_per_cm2"),
                        }
                    )

                df = pd.DataFrame(rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

                # Smooth trends
                df_smooth = df.rolling(window=3, min_periods=1).mean()

                st.line_chart(
                    df_smooth[
                        ["hair_density_score", "scalp_exposed_ratio", "redness_score"]
                    ]
                )

                # Change analysis
                st.markdown("### 📊 Change Since First Scan")
                if len(df) >= 2:
                    baseline = df.iloc[0]
                    latest = df.iloc[-1]

                    col_ch1, col_ch2, col_ch3 = st.columns(3)

                    pct_density = safe_pct_change(
                        baseline["hair_density_score"], latest["hair_density_score"]
                    )
                    pct_exposure = safe_pct_change(
                        baseline["scalp_exposed_ratio"], latest["scalp_exposed_ratio"]
                    )
                    pct_redness = safe_pct_change(
                        baseline["redness_score"], latest["redness_score"]
                    )

                    with col_ch1:
                        if pct_density:
                            st.metric(
                                "Density Change",
                                f"{pct_density:+.1f}%",
                                delta=f"{pct_density:.1f}%",
                                delta_color="normal" if pct_density > 0 else "inverse",
                            )
                        else:
                            st.metric("Density Change", "N/A")

                    with col_ch2:
                        if pct_exposure:
                            st.metric(
                                "Exposure Change",
                                f"{pct_exposure:+.1f}%",
                                delta=f"{pct_exposure:.1f}%",
                                delta_color="inverse" if pct_exposure > 0 else "normal",
                            )
                        else:
                            st.metric("Exposure Change", "N/A")

                    with col_ch3:
                        if pct_redness:
                            st.metric(
                                "Redness Change",
                                f"{pct_redness:+.1f}%",
                                delta=f"{pct_redness:.1f}%",
                                delta_color="inverse" if pct_redness > 0 else "normal",
                            )
                        else:
                            st.metric("Redness Change", "N/A")

            else:
                st.info(
                    "📊 Not enough historical data yet. Complete more scans to see trends over time."
                )
        else:
            st.warning("⚠️ Enter a Tracking ID in the sidebar to enable trend tracking.")

    with tab5:
        st.subheader("📄 Generate PDF Report")

        st.markdown(
            """
            Generate a comprehensive PDF report containing:
            - Analysis summary
            - AI recommendations  
            - Treatment plan
            - Progress tracking guidelines
            """
        )

        if st.button("📥 Generate PDF Report", type="primary"):
            with st.spinner("📝 Generating PDF report..."):
                try:
                    ai_suggestions = results.get("ai_result", {}).get(
                        "ai_suggestions", ""
                    )
                    image_path = results.get("image_path")

                    pdf_buffer = generate_pdf_report(ai_suggestions, image_path)

                    if pdf_buffer and pdf_buffer.getbuffer().nbytes > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="💾 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"hair_analysis_report_{timestamp}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success("✅ PDF report generated successfully!")
                    else:
                        st.error("❌ Failed to generate PDF. Please try again.")

                except Exception as e:
                    st.error(f"❌ PDF generation failed: {str(e)}")
                    import traceback

                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>⚕️ Medical Disclaimer</strong></p>
        <p>This tool provides AI-assisted analysis for educational purposes only.</p>
        <p>Always consult a licensed dermatologist or trichologist for professional medical advice.</p>
        <p style='margin-top: 1rem; font-size: 0.9em;'>
            🔒 Your privacy matters: Data is stored locally. 
            {'✅ Research opt-in enabled' if consent_research else ''}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
