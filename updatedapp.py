import os
import io
import re
import json
import numpy as np
import cv2
import google.generativeai as genai
from markdown import markdown
from datetime import datetime
from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    send_file,
    session,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import black
from PIL import Image as PILImage
import html

# 🔑 Configure Gemini - REPLACE WITH YOUR VALID API KEY
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyDMXotYzFsxDdvYcWU-mYTJK9lxm-SHcmY"
)

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("⚠️  WARNING: Please set your Gemini API key!")
    print(
        "   Option 1: Set environment variable: export GEMINI_API_KEY='your_key_here'"
    )
    print(
        "   Option 2: Replace 'YOUR_GEMINI_API_KEY_HERE' in the code with your actual key"
    )
    print("   Get your key from: https://aistudio.google.com/app/apikey")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")  # Updated to stable model
    print("✅ Gemini API configured successfully")
except Exception as e:
    print(f"❌ Failed to configure Gemini API: {e}")
    model = None

APP_TITLE = "AI Hair Analyzer"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret123")
app.config.update(UPLOAD_FOLDER=UPLOAD_DIR, MAX_CONTENT_LENGTH=8 * 1024 * 1024)


# ========== CV METRICS HELPERS ==========
def estimate_hair_density(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    total_pixels = edges.size
    hair_pixels = np.count_nonzero(edges)
    density_score = hair_pixels / float(total_pixels)

    band = "low"
    if density_score > 0.15:
        band = "medium"
    if density_score > 0.30:
        band = "high"

    scalp_exposed_ratio = 1 - density_score

    return {
        "hair_density_score": round(density_score, 4),
        "hair_density_band": band,
        "scalp_exposed_ratio": round(scalp_exposed_ratio, 4),
    }


def estimate_scalp_redness(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 50, 50])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    redness_score = np.sum(mask > 0) / mask.size

    band = "low"
    if redness_score > 0.1:
        band = "moderate"
    if redness_score > 0.2:
        band = "high"

    return {"redness_score": round(float(redness_score), 4), "redness_band": band}


# ========== AI ANALYSIS FUNCTIONS ==========
def analyze_hair_image(image_path):
    if model is None:
        return {
            "hair_type": "API Error",
            "scalp_condition": "API Error",
            "issues": ["Gemini API not configured"],
        }

    vision_model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """
    You are a dermatologist AI.
    Analyze this scalp/hair photo and return ONLY a JSON object with:
    {
      "hair_type": "Dry | Oily | Normal | Combination",
      "scalp_condition": "Healthy | Oily | Dry | Flaky | Inflamed",
      "issues": ["list of visible issues like dandruff, baldness, fungal infection, hair fall, etc."]
    }
    """

    try:
        with open(image_path, "rb") as img_file:
            response = vision_model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": img_file.read()}]
            )

        ai_text = response.text.strip()
        ai_json = re.search(r"\{.*\}", ai_text, re.DOTALL)
        if ai_json:
            return json.loads(ai_json.group())
        else:
            return {"hair_type": "Unknown", "scalp_condition": "Unknown", "issues": []}
    except Exception as e:
        print(f"Error in analyze_hair_image: {e}")
        return {"hair_type": "Error", "scalp_condition": "Error", "issues": [str(e)]}


def analyze_hair_results(image_path, analysis_results, cv_metrics):
    if model is None:
        return {
            "analysis_summary": "API Configuration Error",
            "ai_suggestions": "⚠️ Gemini API not properly configured. Please check your API key.",
        }

    try:
        prompt = f"""
You are a dermatologist AI.
Analyze the scalp photo, user lifestyle, and CV metrics with a strong focus on **hair loss, baldness, and alopecia**.

Hair Type (AI): {analysis_results.get("hair_type", "Not Provided")}
Scalp Condition (AI): {analysis_results.get("scalp_condition", "Not Provided")}
Detected Issues (AI): {", ".join(analysis_results.get("issues", []))}
Age: {analysis_results.get("age", "Not Provided")}
Sex: {analysis_results.get("sex", "Not Provided")}
Family History: {analysis_results.get("family_history", "Not Provided")}
Stress Level (1–5): {analysis_results.get("stress", "Not Provided")}
Diet Quality (1–5): {analysis_results.get("diet_quality", "Not Provided")}
Sleep Hours: {analysis_results.get("sleep_hours", "Not Provided")}
Regimen Strength: {analysis_results.get("regimen_strength", "Not Provided")}

CV Metrics:
Hair Density Score: {cv_metrics.get("hair_density_score", "N/A")}
Hair Density Band: {cv_metrics.get("hair_density_band", "N/A")}
Scalp Exposed Ratio: {cv_metrics.get("scalp_exposed_ratio", "N/A")}
Redness Score: {cv_metrics.get("redness_score", "N/A")}
Redness Band: {cv_metrics.get("redness_band", "N/A")}

Your output MUST contain:

## Summary
- General overview of scalp and hair health.
- If hair loss, baldness, or alopecia is detected → classify into **Stage 1–7** (use the Norwood scale for male pattern baldness, Ludwig scale if female).
- If no baldness is detected → say "No significant baldness (Stage 0)".

## Factors
- AI detection (hair type, scalp condition, visible bald patches).
- CV metrics (density, scalp exposure, redness).
- Lifestyle factors contributing to hair loss.

## Suggestions
- Non-surgical treatments (shampoos, medications, diet, stress management).
- Preventive measures to slow further hair loss.

## Regimen
   - Daily tips (shampoo/serum, diet focus, scalp massage).
   - Weekly tips (deep conditioning, anti-dandruff care).
   - Monthly tips (checkup, progress tracking).

### Diagnosis & Treatment
1. **Confirmed Diagnosis** (e.g., Stage 2 Hair Loss, Stage 5 Baldness, etc.).
2. **Severity** (Mild / Moderate / Severe).
3. **Treatment Plan**:
   - If baldness is Stage 1–2 → recommend preventive care (non-surgical).
   - If Stage 3–4 → suggest medications + lifestyle improvements, mention surgery option if progression continues.
   - If Stage 5–7 → strongly recommend consultation for **hair transplant surgery**.
   - Clearly explain **possible side effects/risks of transplant** (infection, scarring, unnatural look, shedding phase).
   - Provide **post-transplant prevention tips** (nutrition, stress control, avoiding harsh chemicals).
   - Always suggest when to consult a dermatologist or hair surgeon.

⚠️ Important:
- Only mention **stages for hair loss/baldness/alopecia**.
- Provide at most 5 treatment/product options with clear usage frequency.
- Respond in markdown format for readability.
"""

        with open(image_path, "rb") as img_file:
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": img_file.read()}]
            )

        ai_text = getattr(response, "text", None)

        return {
            "analysis_summary": "AI-generated personalized hair report:",
            "ai_suggestions": (
                ai_text.strip() if ai_text else "⚠️ No suggestions generated."
            ),
        }

    except Exception as e:
        print(f"Error in analyze_hair_results: {e}")
        return {
            "analysis_summary": "Error",
            "ai_suggestions": f"⚠️ AI analysis failed: {str(e)}",
        }


# ========== HELPER FUNCTIONS FOR PDF ==========
def clean_markdown_for_pdf(text):
    """Clean markdown text for PDF generation"""
    # Remove complex markdown elements that cause issues
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)  # Bold
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)  # Italic
    text = re.sub(r"#{1,6}\s*(.*?)(?:\n|$)", r"<b>\1</b><br/>", text)  # Headers
    text = re.sub(r"\n\n+", "<br/><br/>", text)  # Multiple newlines
    text = re.sub(r"\n", "<br/>", text)  # Single newlines

    # Escape any remaining HTML
    text = html.unescape(text)

    return text


def resize_image_for_pdf(image_path, max_width=400, max_height=300):
    """Resize image to fit in PDF"""
    try:
        with PILImage.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            # Calculate new dimensions
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height

            if img_width > max_width or img_height > max_height:
                if aspect_ratio > 1:  # Landscape
                    new_width = max_width
                    new_height = int(max_width / aspect_ratio)
                else:  # Portrait
                    new_height = max_height
                    new_width = int(max_height * aspect_ratio)

                img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

            # Save temporary resized image
            temp_path = image_path.replace(os.path.splitext(image_path)[1], "_temp.jpg")
            img.save(temp_path, "JPEG", quality=85)
            return temp_path, img.size
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None, (0, 0)


# ========== PDF GENERATOR ==========
def generate_pdf_report(ai_suggestions, scalp_image_path=None):
    try:
        buffer = io.BytesIO()

        def add_footer(canvas_obj, doc):
            try:
                canvas_obj.saveState()
                footer_text = f"Generated by AI Hair Analyzer – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                canvas_obj.setFont("Helvetica-Oblique", 8)
                # Use drawCentredText or drawString based on ReportLab version
                try:
                    canvas_obj.drawCentredText(A4[0] / 2.0, 0.5 * inch, footer_text)
                except AttributeError:
                    # Fallback for newer ReportLab versions
                    text_width = canvas_obj.stringWidth(
                        footer_text, "Helvetica-Oblique", 8
                    )
                    canvas_obj.drawString(
                        (A4[0] - text_width) / 2.0, 0.5 * inch, footer_text
                    )
                canvas_obj.restoreState()
            except Exception as e:
                print(f"Error in footer: {e}")
                pass

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()

        # Create custom styles with error handling
        try:
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Title"],
                fontSize=20,
                spaceAfter=20,
                alignment=1,  # Center alignment
            )
        except:
            title_style = styles["Title"]

        try:
            normal_style = ParagraphStyle(
                "CustomNormal",
                parent=styles["Normal"],
                fontSize=11,
                spaceAfter=6,
                leftIndent=0,
                rightIndent=0,
            )
        except:
            normal_style = styles["Normal"]

        story = []

        # Add title
        try:
            story.append(Paragraph("AI Hair Analysis Report", title_style))
            story.append(Spacer(1, 20))
        except Exception as e:
            print(f"Error adding title: {e}")
            story.append(Paragraph("AI Hair Analysis Report", styles["Title"]))
            story.append(Spacer(1, 20))

        # Add image if provided
        temp_image_path = None
        if scalp_image_path and os.path.exists(scalp_image_path):
            try:
                temp_image_path, (img_width, img_height) = resize_image_for_pdf(
                    scalp_image_path
                )
                if temp_image_path and os.path.exists(temp_image_path):
                    # Add image with proper sizing
                    img = ReportLabImage(
                        temp_image_path, width=img_width, height=img_height
                    )
                    story.append(img)
                    story.append(Spacer(1, 20))
            except Exception as e:
                print(f"Error adding image to PDF: {e}")
                try:
                    story.append(
                        Paragraph(
                            f"[Scalp image uploaded but could not be displayed in PDF]",
                            normal_style,
                        )
                    )
                    story.append(Spacer(1, 12))
                except:
                    pass

        # Process content with better error handling
        try:
            if ai_suggestions and ai_suggestions.strip():
                # Simple text processing without complex markdown
                lines = (
                    ai_suggestions.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                )

                current_paragraph = ""
                for line in lines:
                    line = line.strip()
                    if not line:
                        # Empty line - end current paragraph
                        if current_paragraph:
                            try:
                                # Clean the paragraph text
                                safe_text = (
                                    html.escape(current_paragraph)
                                    .replace("&lt;", "<")
                                    .replace("&gt;", ">")
                                )
                                p = Paragraph(safe_text, normal_style)
                                story.append(p)
                                story.append(Spacer(1, 8))
                            except Exception as pe:
                                print(f"Error adding paragraph: {pe}")
                                # Ultra-safe fallback
                                try:
                                    safe_text = current_paragraph.replace(
                                        "<", "&lt;"
                                    ).replace(">", "&gt;")
                                    p = Paragraph(safe_text, styles["Normal"])
                                    story.append(p)
                                    story.append(Spacer(1, 8))
                                except:
                                    pass
                            current_paragraph = ""
                    else:
                        # Add line to current paragraph
                        if current_paragraph:
                            current_paragraph += " " + line
                        else:
                            current_paragraph = line

                # Add final paragraph if exists
                if current_paragraph:
                    try:
                        safe_text = (
                            html.escape(current_paragraph)
                            .replace("&lt;", "<")
                            .replace("&gt;", ">")
                        )
                        p = Paragraph(safe_text, normal_style)
                        story.append(p)
                    except Exception as pe:
                        print(f"Error adding final paragraph: {pe}")
                        try:
                            safe_text = current_paragraph.replace("<", "&lt;").replace(
                                ">", "&gt;"
                            )
                            p = Paragraph(safe_text, styles["Normal"])
                            story.append(p)
                        except:
                            pass
            else:
                story.append(Paragraph("No analysis content available.", normal_style))

        except Exception as e:
            print(f"Error processing content: {e}")
            # Ultra-safe fallback
            try:
                story.append(
                    Paragraph("Hair analysis completed successfully.", styles["Normal"])
                )
                story.append(Spacer(1, 12))
                story.append(
                    Paragraph(
                        "Content could not be formatted for PDF display.",
                        styles["Normal"],
                    )
                )
            except:
                pass

        # Build PDF with error handling
        try:
            doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
        except Exception as e:
            print(f"Error building PDF with footer: {e}")
            # Try without footer
            try:
                doc.build(story)
            except Exception as e2:
                print(f"Error building PDF without footer: {e2}")
                # Create minimal PDF
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                minimal_story = [
                    Paragraph("AI Hair Analysis Report", styles["Title"]),
                    Spacer(1, 20),
                    Paragraph("Analysis completed successfully.", styles["Normal"]),
                    Spacer(1, 12),
                    Paragraph(
                        "Please contact support if you need the detailed report.",
                        styles["Normal"],
                    ),
                ]
                doc.build(minimal_story)

        # Clean up temporary image
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up temp image: {cleanup_error}")

        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Major error generating PDF: {e}")
        # Return absolute minimal PDF
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("AI Hair Analysis Report", styles["Title"]),
                Spacer(1, 20),
                Paragraph(
                    "Analysis completed but PDF formatting encountered technical issues.",
                    styles["Normal"],
                ),
                Spacer(1, 12),
                Paragraph(
                    "Please try downloading again or contact support.", styles["Normal"]
                ),
            ]
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as final_error:
            print(f"Final PDF generation error: {final_error}")
            # Return empty buffer as last resort
            return io.BytesIO()


# ========== UI ==========
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ APP_TITLE }}</title>
  <style>
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      margin: 2rem;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      border-radius: 20px;
      padding: 2rem;
      box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    h1 {
      text-align: center;
      color: #2d3748;
      margin-bottom: 0.5rem;
    }
    .subtitle {
      text-align: center;
      color: #718096;
      margin-bottom: 2rem;
    }
    .card {
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 20px rgba(0,0,0,0.08);
      background: white;
    }
    .grid {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }
    label {
      display: block;
      font-weight: 600;
      margin: .5rem 0 .25rem;
      color: #374151;
    }
    input, select {
      width: 100%;
      padding: .8rem .9rem;
      border-radius: 10px;
      border: 2px solid #e5e7eb;
      transition: border-color 0.3s;
      box-sizing: border-box;
    }
    input:focus, select:focus {
      outline: none;
      border-color: #667eea;
    }
    button {
      padding: 1rem 2rem;
      border-radius: 12px;
      border: 0;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s;
      width: 100%;
      margin-top: 1rem;
    }
    button:hover {
      transform: translateY(-2px);
    }
    .results {
      line-height: 1.6;
    }
    .error {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #dc2626;
      padding: 1rem;
      border-radius: 8px;
    }
    .api-warning {
      background: #fffbeb;
      border: 1px solid #fed7aa;
      color: #d97706;
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 1rem;
    }
    .success {
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 1rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>{{ APP_TITLE }}</h1>
    <p class="subtitle">Upload a scalp photo and provide details for AI-powered hair analysis</p>

    {% if api_warning %}
    <div class="api-warning">
      <strong>⚠️ API Configuration Warning:</strong><br>
      {{ api_warning|safe }}
    </div>
    {% endif %}

    {% if success_message %}
    <div class="success">
      <strong>✅ Success:</strong> {{ success_message }}
    </div>
    {% endif %}

    <form class="card" action="/analyze" method="post" enctype="multipart/form-data">
      <div class="grid">
        <div>
          <label>📸 Scalp Photo</label>
          <input type="file" name="photo" accept="image/*" required />
        </div>
        <div>
          <label>👤 Age</label>
          <input type="number" name="age" value="28" min="1" max="100" required />
        </div>
        <div>
          <label>⚧ Sex</label>
          <select name="sex">
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
        <div>
          <label>🧬 Family History of Hair Loss</label>
          <select name="family_history">
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
        </div>
        <div>
          <label>😰 Stress Level (1-5)</label>
          <input type="number" name="stress" min="1" max="5" value="3" />
        </div>
        <div>
          <label>🥗 Diet Quality (1-5)</label>
          <input type="number" name="diet_quality" min="1" max="5" value="3" />
        </div>
        <div>
          <label>😴 Sleep Hours per Night</label>
          <input type="number" step="0.5" name="sleep_hours" value="7" min="4" max="12" />
        </div>
        <div>
          <label>💊 Current Hair Care Regimen</label>
          <select name="regimen_strength">
            <option value="none">None</option>
            <option value="standard" selected>Standard</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>
      </div>
      <button type="submit">🔬 Analyze Hair Health</button>
    </form>

    {% if result %}
    <div class="card">
      <h2>🤖 AI Analysis Results</h2>
      <h3>{{ result.analysis_summary }}</h3>
      <div class="results">{{ result.ai_html | safe }}</div>
      
      {% if result.cv_metrics %}
      <div style="margin-top: 2rem; padding: 1rem; background: #f8fafc; border-radius: 8px;">
        <h4>📊 Computer Vision Metrics:</h4>
        <p><strong>Hair Density:</strong> {{ result.cv_metrics.hair_density_band }} ({{ result.cv_metrics.hair_density_score }})</p>
        <p><strong>Scalp Exposure:</strong> {{ (result.cv_metrics.scalp_exposed_ratio * 100) | round(1) }}%</p>
        <p><strong>Scalp Redness:</strong> {{ result.cv_metrics.redness_band }} ({{ result.cv_metrics.redness_score }})</p>
      </div>
      {% endif %}
      
      <a href="/download"><button>📄 Download PDF Report</button></a>
    </div>
    {% endif %}

    {% if error %}
    <div class="error">
      <strong>❌ Error:</strong> {{ error }}
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


# ========== ROUTES ==========
@app.route("/", methods=["GET"])
def index():
    api_warning = None
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        api_warning = """
        Please configure your Gemini API key:<br>
        1. Visit <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a><br>
        2. Create an API key<br>
        3. Set environment variable: <code>export GEMINI_API_KEY='your_key_here'</code><br>
        4. Or replace the placeholder in the code
        """

    success_message = session.pop("success_message", None)
    return render_template_string(
        INDEX_HTML,
        APP_TITLE=APP_TITLE,
        api_warning=api_warning,
        success_message=success_message,
    )


@app.route("/analyze", methods=["POST"])
def analyze_form():
    file = request.files.get("photo")
    if not file:
        return render_template_string(
            INDEX_HTML, result=None, error="No file uploaded", APP_TITLE=APP_TITLE
        )

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        image_path = os.path.join(UPLOAD_DIR, filename)
        file.save(image_path)

        # Validate image
        img = cv2.imread(image_path)
        if img is None:
            return render_template_string(
                INDEX_HTML, result=None, error="Invalid image file", APP_TITLE=APP_TITLE
            )

        # Collect user form data
        age = int(request.form.get("age", 28))
        sex = request.form.get("sex", "male")
        family_history = request.form.get("family_history", "no") == "yes"
        stress = int(request.form.get("stress", 3))
        diet_quality = int(request.form.get("diet_quality", 3))
        sleep_hours = float(request.form.get("sleep_hours", 7))
        regimen_strength = request.form.get("regimen_strength", "standard")

        # AI vision analysis
        ai_analysis = analyze_hair_image(image_path)

        # CV metrics
        cv_metrics = {}
        cv_metrics.update(estimate_hair_density(img))
        cv_metrics.update(estimate_scalp_redness(img))

        # Merge AI + user inputs + CV
        result = analyze_hair_results(
            image_path,
            {
                **ai_analysis,
                "age": age,
                "sex": sex,
                "family_history": family_history,
                "stress": stress,
                "diet_quality": diet_quality,
                "sleep_hours": sleep_hours,
                "regimen_strength": regimen_strength,
            },
            cv_metrics,
        )

        # Render markdown
        result["ai_html"] = markdown(result["ai_suggestions"])
        result["cv_metrics"] = cv_metrics

        session["last_result"] = result["ai_suggestions"]  # Store only the text
        session["last_image"] = image_path

        return render_template_string(INDEX_HTML, result=result, APP_TITLE=APP_TITLE)

    except Exception as e:
        print(f"Error in analyze_form: {e}")
        return render_template_string(
            INDEX_HTML, result=None, error=str(e), APP_TITLE=APP_TITLE
        )


@app.route("/download", methods=["GET"])
def download_report():
    result_text = session.get("last_result")
    image_path = session.get("last_image")

    if not result_text:
        session["success_message"] = (
            "No analysis results found. Please analyze an image first."
        )
        return redirect(url_for("index"))

    try:
        print(
            f"Generating PDF with result_text length: {len(result_text) if result_text else 0}"
        )
        print(f"Image path: {image_path}")

        pdf_buffer = generate_pdf_report(result_text, image_path)

        if pdf_buffer and pdf_buffer.getvalue():
            pdf_buffer.seek(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=f"Hair_Analysis_Report_{timestamp}.pdf",
                mimetype="application/pdf",
            )
        else:
            session["success_message"] = "Error: Empty PDF generated"
            return redirect(url_for("index"))

    except Exception as e:
        print(f"Error in download_report: {e}")
        import traceback

        traceback.print_exc()
        session["success_message"] = f"Error generating PDF: {str(e)}"
        return redirect(url_for("index"))


# ========== HEALTH CHECK ==========
@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "healthy",
        "gemini_api": "configured" if model is not None else "not_configured",
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(status)


# ========== RUN APP ==========
if __name__ == "__main__":
    print("🚀 Starting AI Hair Analyzer...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"🔑 API Status: {'✅ Configured' if model else '❌ Not Configured'}")
    app.run(host="0.0.0.0", port=5000, debug=True)
