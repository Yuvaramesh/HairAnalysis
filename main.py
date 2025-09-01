import os
import io
import re
import json
import numpy as np
import cv2
import google.generativeai as genai
import markdown
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_file, session, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ----------------------------
# NEW: YOLOv8 (Ultralytics)
# ----------------------------
from ultralytics import YOLO

# =============== Gemini Setup ===============
genai.configure(api_key="AIzaSyAn26uG2YjfKgD7b9B0td39KxcmdSQZZ48")
model = genai.GenerativeModel("gemini-2.5-flash")

# =============== App & Paths ===============
APP_TITLE = "AI Hair Analyzer (Demo)"
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret123")  # needed for session
app.config.update(UPLOAD_FOLDER=UPLOAD_DIR, MAX_CONTENT_LENGTH=8 * 1024 * 1024)

# ----------------------------
# NEW: Load YOLO model
# - Replace with your custom hair model: "runs/segment/train/weights/best.pt"
#   or any custom .pt you trained for dandruff/baldness/hair-seg.
# ----------------------------
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")  # demo weights
yolo_model = YOLO(YOLO_WEIGHTS)

# Utility: build an annotated filename
def _annotated_name(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_det{ext}"

# Utility: validate image extension
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Utility Functions
def _read_image(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img

# ----------------------------
# NEW: YOLO inference helper
# ----------------------------
def detect_scalp_issues(image_path: str):
    """
    Run YOLO on image_path. Returns:
      - detections: list of {label, confidence}
      - annotated_path: saved annotated image path
    Notes:
      * With default COCO weights, labels are generic (person, etc).
      * When you plug a custom hair model, labels e.g. 'hair', 'dandruff', 'bald_patch'
        will come from your model's .pt.
    """
    results = yolo_model(image_path)  # inference
    detections = []

    # take first result frame
    r = results[0]

    # Collect detection info (boxes)
    if r.boxes is not None:
        names = r.names  # class id -> name
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            detections.append({"label": label, "confidence": round(conf, 2)})

    # If segmentation masks available (when using seg model), r.masks exists.
    # We still just render the standard annotated image for simplicity.

    # Save annotated image
    annotated = r.plot()  # returns a BGR numpy array with drawings
    out_path = _annotated_name(image_path)
    cv2.imwrite(out_path, annotated)

    return detections, out_path

def analyze_hair_results(analysis_results):
    """Call Gemini API for AI-based suggestions, taking YOLO issues into account."""
    # Build issues nicely for prompt
    issues_list = analysis_results.get("issues", [])
    issues_text = ", ".join(issues_list) if issues_list else "None"

    prompt = f"""
    You are a professional trichologist AI.
    Based on the following hair analysis results and computer-vision detections,
    provide a clear summary and actionable personalized suggestions.

    Hair Type: {analysis_results.get("hair_type", "Not specified")}
    Scalp Condition: {analysis_results.get("scalp_condition", "Not specified")}
    CV Detected Issues: {issues_text}
    Age: {analysis_results.get("age")}
    Sex: {analysis_results.get("sex")}
    Family History: {analysis_results.get("family_history")}
    Stress Level (1-5): {analysis_results.get("stress")}
    Diet Quality (1-5): {analysis_results.get("diet_quality")}
    Sleep Hours: {analysis_results.get("sleep_hours")}
    Regimen Strength: {analysis_results.get("regimen_strength")}

    IMPORTANT:
    - If issues look generic (e.g., from a non-hair-pretrained model), say so briefly.
    - If bald patches or dandruff are detected, quantify likely severity qualitatively (mild/moderate/severe).
    - Suggest lifestyle, diet, scalp care routine, and product types.
    - Keep it crisp and scannable.

    Format response with:
    ## Summary of Condition
    ## AI Recommendations
    - Lifestyle
    - Diet
    - Scalp Care & Products
    - When to see a dermatologist
    """

    try:
        response = model.generate_content(prompt)
        return {
            "analysis_summary": "AI-generated personalized hair report:",
            "ai_suggestions": response.text.strip() if response.text else "No suggestions generated."
        }
    except Exception as e:
        return {"analysis_summary": "Error", "ai_suggestions": str(e)}

def generate_pdf_report(ai_suggestions, scalp_image_path=None):
    """Generate PDF report with scalp image + AI suggestions (Markdown-aware + footer)"""
    buffer = io.BytesIO()

    # Footer function
    def add_footer(canvas_obj, doc):
        canvas_obj.saveState()
        footer_text = f"Generated by AI Hair Analyzer – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        canvas_obj.setFont("Helvetica-Oblique", 8)
        canvas_obj.drawCentredString(A4[0] / 2.0, 0.5 * inch, footer_text)
        canvas_obj.restoreState()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = styles["Title"]
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], spaceAfter=8, textColor="#2E3A59")
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], leftIndent=20, bulletIndent=10, spaceAfter=4)
    normal_style = styles["Normal"]

    story = []
    story.append(Paragraph("AI Hair Analysis Report", title_style))
    story.append(Spacer(1, 12))

    if scalp_image_path and os.path.exists(scalp_image_path):
        # Keep width in bounds; height auto-ish
        story.append(Image(scalp_image_path, width=300, height=200))
        story.append(Spacer(1, 12))

    # Convert Gemini markdown output into PDF-friendly blocks
    lines = ai_suggestions.split("\n")
    bullets = []
    for line in lines:
        line = line.strip()
        if not line:
            if bullets:
                story.append(ListFlowable(bullets, bulletType="bullet", start="•", leftIndent=20))
                story.append(Spacer(1, 8))
                bullets = []
            continue

        if line.startswith("##"):  # Heading
            if bullets:
                story.append(ListFlowable(bullets, bulletType="bullet", start="•", leftIndent=20))
                bullets = []
            story.append(Paragraph(line.replace("##", "").strip(), heading_style))
            story.append(Spacer(1, 6))

        elif line.startswith("* ") or line.startswith("- "):  # Bullet list
            bullets.append(ListItem(Paragraph(line[2:], bullet_style)))

        elif re.match(r"^\d+\.", line):  # Numbered list
            story.append(Paragraph(line, normal_style))

        elif line.startswith("**") and line.endswith("**"):  # Bold line
            if bullets:
                story.append(ListFlowable(bullets, bulletType="bullet", start="•", leftIndent=20))
                bullets = []
            story.append(Paragraph(f"<b>{line.strip('*')}</b>", normal_style))
            story.append(Spacer(1, 6))

        elif "**" in line:  # Bold phrase inside text
            formatted = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            story.append(Paragraph(formatted, normal_style))

        else:  # Normal text
            story.append(Paragraph(line, normal_style))

    # Flush any remaining bullets
    if bullets:
        story.append(ListFlowable(bullets, bulletType="bullet", start="•", leftIndent=20))

    # Build doc with footer callback
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)

    buffer.seek(0)
    return buffer

# HTML Template (added preview of annotated image + raw YOLO detections)
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ APP_TITLE }}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
    .card { border: 1px solid #e5e7eb; border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 10px rgba(0,0,0,.04); }
    .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    label { display:block; font-weight:600; margin:.5rem 0 .25rem; }
    input, select { width:100%; padding:.6rem .7rem; border-radius:10px; border:1px solid #d1d5db; }
    button { padding:.8rem 1rem; border-radius:12px; border:0; background:#111827; color:white; font-weight:600; cursor:pointer; }
    .results { line-height:1.6; }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #e5e7eb; }
    code { background: #f6f7f9; padding: .15rem .35rem; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>{{ APP_TITLE }}</h1>
  <p>Upload a scalp photo and provide details below.</p>

  <form class="card" action="/analyze" method="post" enctype="multipart/form-data">
    <div class="grid">
      <div><label>Scalp Photo</label><input type="file" name="photo" accept="image/*" required /></div>
      <div><label>Age</label><input type="number" name="age" value="28" required /></div>
      <div><label>Sex</label><select name="sex"><option>male</option><option>female</option></select></div>
      <div><label>Family History</label><select name="family_history"><option value="no">No</option><option value="yes">Yes</option></select></div>
      <div><label>Stress (1–5)</label><input type="number" name="stress" min="1" max="5" value="3"></div>
      <div><label>Diet Quality (1–5)</label><input type="number" name="diet_quality" min="1" max="5" value="3"></div>
      <div><label>Sleep Hours</label><input type="number" step="0.1" name="sleep_hours" value="7"></div>
      <div><label>Regimen</label><select name="regimen_strength"><option>standard</option><option>none</option><option>aggressive</option></select></div>
    </div>
    <button type="submit">Analyze</button>
  </form>

  {% if result %}
  <div class="card">
    <h2>Computer-Vision Detections (YOLO)</h2>
    {% if result.yolo_detections and result.yolo_detections|length > 0 %}
      <ul>
        {% for d in result.yolo_detections %}
          <li><code>{{ d.label }}</code> — {{ d.confidence }}</li>
        {% endfor %}
      </ul>
    {% else %}
      <p>No specific scalp issues detected by the current model (or generic model used).</p>
    {% endif %}

    {% if result.annotated_rel %}
      <h3>Annotated Image</h3>
      <img src="{{ url_for('serve_upload', filename=result.annotated_rel) }}" alt="Annotated scalp image" />
    {% endif %}
  </div>

  <div class="card">
    <h2>AI Results</h2>
    <h3>{{ result.analysis_summary }}</h3>
    <div class="results">{{ result.ai_html | safe }}</div>
    <a href="/download"><button>Download PDF Report</button></a>
  </div>
  {% endif %}

  {% if error %}
  <div class="card"><b>Error:</b> {{ error }}</div>
  {% endif %}
</body>
</html>
"""

# Static serving for uploaded / annotated images
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# Routes
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
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return render_template_string(INDEX_HTML, result=None, error="Unsupported file type", APP_TITLE=APP_TITLE)

        image_path = os.path.join(UPLOAD_DIR, filename)
        file.save(image_path)

        # collect form data
        age = int(request.form.get("age", 28))
        sex = request.form.get("sex", "male")
        family_history = request.form.get("family_history", "no") == "yes"
        stress = int(request.form.get("stress", 3))
        diet_quality = int(request.form.get("diet_quality", 3))
        sleep_hours = float(request.form.get("sleep_hours", 7))
        regimen_strength = request.form.get("regimen_strength", "standard")

        # ----------------------------
        # NEW: Run YOLO and get detections
        # ----------------------------
        yolo_detections, annotated_path = detect_scalp_issues(image_path)

        # Build "issues" for Gemini prompt from YOLO labels
        issues = [d["label"] for d in yolo_detections]

        # Call Gemini with both user context + YOLO detections
        result = analyze_hair_results({
            "hair_type": "Dry",            # placeholder; you can infer via rules if needed
            "scalp_condition": "Oily",     # placeholder
            "issues": issues,              # YOLO -> Gemini
            "age": age,
            "sex": sex,
            "family_history": family_history,
            "stress": stress,
            "diet_quality": diet_quality,
            "sleep_hours": sleep_hours,
            "regimen_strength": regimen_strength,
        })

        # Render markdown for UI
        result["ai_html"] = markdown.markdown(result["ai_suggestions"])

        # Extra: include raw yolo detections + annotated image (relative name) for HTML
        annotated_rel = os.path.basename(annotated_path)
        result["yolo_detections"] = yolo_detections
        result["annotated_rel"] = annotated_rel

        # Store JSON-safe in session (store ai text + keep image absolute path separately)
        session["last_result"] = json.dumps({
            "analysis_summary": result["analysis_summary"],
            "ai_suggestions": result["ai_suggestions"]
        })
        session["last_image"] = annotated_path  # PDF will embed annotated

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

    return send_file(pdf_buffer, as_attachment=True,
                     download_name="Hair_Report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    # Helpful log
    print(f"[INFO] Using YOLO weights: {YOLO_WEIGHTS}")
    app.run(host="0.0.0.0", port=5000, debug=True)
