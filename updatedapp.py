import os
import io
import re
import json
import numpy as np
import cv2
import google.generativeai as genai
from markdown import markdown  
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# 🔑 Configure Gemini
genai.configure(api_key="AIzaSyAn26uG2YjfKgD7b9B0td39KxcmdSQZZ48")
model = genai.GenerativeModel("gemini-2.5-flash")

APP_TITLE = "AI Hair Analyzer (Demo)"
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
        "scalp_exposed_ratio": round(scalp_exposed_ratio, 4)
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

    return {
        "redness_score": round(float(redness_score), 4),
        "redness_band": band
    }

# ========== AI ANALYSIS FUNCTIONS ==========
def analyze_hair_image(image_path):
    vision_model = genai.GenerativeModel("gemini-2.5-flash")

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
        return {"hair_type": "Error", "scalp_condition": "Error", "issues": [str(e)]}

def analyze_hair_results(image_path, analysis_results, cv_metrics):
    try:
        prompt = f"""
You are a dermatologist AI.
Analyze the scalp photo, user lifestyle, and CV metrics.

Hair Type (AI): {analysis_results.get("hair_type")}
Scalp Condition (AI): {analysis_results.get("scalp_condition")}
Detected Issues (AI): {", ".join(analysis_results.get("issues", []))}
Age: {analysis_results.get("age")}
Sex: {analysis_results.get("sex")}
Family History: {analysis_results.get("family_history")}
Stress Level (1–5): {analysis_results.get("stress")}
Diet Quality (1–5): {analysis_results.get("diet_quality")}
Sleep Hours: {analysis_results.get("sleep_hours")}
Regimen Strength: {analysis_results.get("regimen_strength")}

CV Metrics:
Hair Density Score: {cv_metrics['hair_density_score']}
Hair Density Band: {cv_metrics['hair_density_band']}
Scalp Exposed Ratio: {cv_metrics['scalp_exposed_ratio']}
Redness Score: {cv_metrics['redness_score']}
Redness Band: {cv_metrics['redness_band']}

Your output MUST contain:
## Summary
## Factors (AI detection + CV metrics + lifestyle inputs)
## Suggestions
## Regimen
   - Daily tips
   - Weekly tips
   - Monthly tips
### Diagnosis & Treatment
1. **Confirmed Diagnosis** (single best guess).
2. **Severity** (Mild / Moderate / Severe).
3. **Treatment Plan**:
   - Primary shampoo/medication (1–2 options max, with usage frequency).
   - Lifestyle tips (stress, diet, sleep).
   - When to see a dermatologist.

Do not provide more than 5 product options.
Always mention usage frequency.
Respond in markdown format.
"""

        with open(image_path, "rb") as img_file:
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": img_file.read()}]
            )

        return {
            "analysis_summary": "AI-generated personalized hair report:",
            "ai_suggestions": response.text.strip() if response.text else "No suggestions generated."
        }
    except Exception as e:
        return {"analysis_summary": "Error", "ai_suggestions": str(e)}

# ========== PDF GENERATOR ==========
def generate_pdf_report(ai_suggestions, scalp_image_path=None):
    buffer = io.BytesIO()

    def add_footer(canvas_obj, doc):
        canvas_obj.saveState()
        footer_text = f"Generated by AI Hair Analyzer – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        canvas_obj.setFont("Helvetica-Oblique", 8)
        canvas_obj.drawCentredString(A4[0] / 2.0, 0.5 * inch, footer_text)
        canvas_obj.restoreState()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Hair Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))

    if scalp_image_path and os.path.exists(scalp_image_path):
        story.append(Image(scalp_image_path, width=250, height=180))
        story.append(Spacer(1, 12))

    html_content = markdown(ai_suggestions)
    for paragraph in html_content.split("\n"):
        if paragraph.strip():
            story.append(Paragraph(paragraph, styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer
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
    }
    .card {
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 1.25rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 10px rgba(0,0,0,.04);
    }
    .grid {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }
    label {
      display: block;
      font-weight: 600;
      margin: .5rem 0 .25rem;
    }
    input, select {
      width: 100%;
      padding: .6rem .7rem;
      border-radius: 10px;
      border: 1px solid #d1d5db;
    }
    button {
      padding: .8rem 1rem;
      border-radius: 12px;
      border: 0;
      background: #111827;
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    .results {
      line-height: 1.6;
    }
  </style>
</head>
<body>
  <h1>{{ APP_TITLE }}</h1>
  <p>Upload a scalp photo and provide details below.</p>

  <form class="card" action="/analyze" method="post" enctype="multipart/form-data">
    <div class="grid">
      <div>
        <label>Scalp Photo</label>
        <input type="file" name="photo" accept="image/*" required />
      </div>
      <div>
        <label>Age</label>
        <input type="number" name="age" value="28" required />
      </div>
      <div>
        <label>Sex</label>
        <select name="sex">
          <option>male</option>
          <option>female</option>
        </select>
      </div>
      <div>
        <label>Family History</label>
        <select name="family_history">
          <option value="no">No</option>
          <option value="yes">Yes</option>
        </select>
      </div>
      <div>
        <label>Stress (1–5)</label>
        <input type="number" name="stress" min="1" max="5" value="3" />
      </div>
      <div>
        <label>Diet Quality (1–5)</label>
        <input type="number" name="diet_quality" min="1" max="5" value="3" />
      </div>
      <div>
        <label>Sleep Hours</label>
        <input type="number" step="0.1" name="sleep_hours" value="7" />
      </div>
      <div>
        <label>Regimen</label>
        <select name="regimen_strength">
          <option>standard</option>
          <option>none</option>
          <option>aggressive</option>
        </select>
      </div>
    </div>
    <button type="submit">Analyze</button>
  </form>

  {% if result %}
  <div class="card">
    <h2>AI Results</h2>
    <h3>{{ result.analysis_summary }}</h3>
    <div class="results">{{ result.ai_html | safe }}</div>
    <a href="/download"><button>Download PDF Report</button></a>
  </div>
  {% endif %}

  {% if error %}
  <div class="card">
    <b>Error:</b> {{ error }}
  </div>
  {% endif %}
</body>
</html>
"""

# ========== ROUTES ==========
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, APP_TITLE=APP_TITLE)

@app.route("/analyze", methods=["POST"])
def analyze_form():
    file = request.files.get("photo")
    if not file:
        return render_template_string(INDEX_HTML, result=None, error="No file uploaded", APP_TITLE=APP_TITLE)

    try:
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_DIR, filename)
        file.save(image_path)

        img = cv2.imread(image_path)

        # collect user form data
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
        if img is not None:
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
            cv_metrics
        )

        # Render markdown
        result["ai_html"] = markdown(result["ai_suggestions"])
        result["cv_metrics"] = cv_metrics

        session["last_result"] = json.dumps(result)
        session["last_image"] = image_path

        return render_template_string(INDEX_HTML, result=result, APP_TITLE=APP_TITLE)

    except Exception as e:
        return render_template_string(INDEX_HTML, result=None, error=str(e), APP_TITLE=APP_TITLE)

@app.route("/download", methods=["GET"])
def download_report():
    result_json = session.get("last_result")
    image_path = session.get("last_image")

    if not result_json:
        return redirect(url_for("index"))

    result = json.loads(result_json)
    pdf_buffer = generate_pdf_report(result["ai_suggestions"], image_path)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="Hair_Report.pdf",
        mimetype="application/pdf"
    )

# ========== RUN APP ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
