from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import pandas as pd
import PyPDF2
import io
import os
import glob

# utils
from utils.skill_extraction import extract_skills

# Get the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Files in base dir: {os.listdir(BASE_DIR) if BASE_DIR.exists() else 'N/A'}")

# Find the data file 
data_file_path = BASE_DIR / "data" / "jobs_dataset.csv"
if not data_file_path.exists():
    # Try to find it recursively
    data_files = glob.glob(f"{BASE_DIR}/**/jobs_dataset.csv", recursive=True)
    if data_files:
        data_file_path = data_files[0]
        print(f"Found data file: {data_file_path}")
    else:
        print(f"Data file not found at {data_file_path}")
else:
    print(f"Using data file: {data_file_path}")

app = FastAPI()

# =========================
# 📂 TEMPLATES + STATIC
# =========================
# Use absolute paths based on BASE_DIR
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

print(f"Templates dir: {templates_dir} (exists: {templates_dir.exists()})")
print(f"Static dir: {static_dir} (exists: {static_dir.exists()})")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    print("Jinja2Templates initialized successfully")
else:
    print(f"ERROR: Templates directory not found at {templates_dir}")
    raise FileNotFoundError(f"Templates directory not found at {templates_dir}")

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print("Static files mounted successfully")
else:
    print(f"ERROR: Static directory not found at {static_dir}")


def render_index(**context):
    template = templates.env.get_template("index.html")
    return HTMLResponse(template.render(**context))

# =========================
# 📂 LOAD DATASET
# =========================
try:
    df = pd.read_csv(str(data_file_path))
    df.columns = df.columns.str.strip().str.lower()
    print(f"Dataset loaded successfully: {len(df)} rows")
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempted data file path: {data_file_path}")
    print(f"Available CSV files: {glob.glob(f'{BASE_DIR}/**/*.csv', recursive=True)}")
    raise

# =========================
# 🔍 RECOMMENDATION LOGIC
# =========================
def recommend_logic(user_skills: List[str], top_n: int = 5):
    recommendations = []

    for _, row in df.iterrows():
        job_skills = str(row.get('required_skills', '')).lower()

        score = sum(
            1 for skill in user_skills if skill.lower() in job_skills
        )

        recommendations.append({
            "job_role": row.get("job_role", "Unknown"),
            "category": row.get("skill_category", "Unknown"),
            "score": score
        })

    recommendations = sorted(
        recommendations,
        key=lambda x: x['score'],
        reverse=True
    )

    return recommendations[:top_n]

# =========================
# 🏠 HOME PAGE
# =========================
@app.get("/")
def home(request: Request):
    try:
        print("[DEBUG] Home route called")
        print(f"[DEBUG] Templates directory: {templates_dir}")
        response = render_index()
        print("[DEBUG] Template rendered successfully")
        return response
    except Exception as e:
        print(f"[ERROR] In home route: {e}")
        import traceback
        print(traceback.format_exc())
        raise


@app.get("/debug")
def debug():
    """Debug endpoint to check configuration"""
    return {
        "base_dir": str(BASE_DIR),
        "templates_dir": str(templates_dir),
        "templates_exists": templates_dir.exists(),
        "static_dir": str(static_dir),
        "static_exists": static_dir.exists(),
        "data_file": str(data_file_path),
        "data_exists": data_file_path.exists() if isinstance(data_file_path, Path) else False,
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = BASE_DIR / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    return Response(status_code=204)

# =========================
# 📡 ANALYZE (TEXT + FILE)
# =========================
@app.post("/analyze")
async def analyze_form(
    request: Request,
    resume_text: str = Form(None),
    file: UploadFile = File(None)
):

    text = ""

    # 📄 FILE HANDLE
    if file:
        if file.filename.endswith(".pdf"):
            pdf = PyPDF2.PdfReader(io.BytesIO(await file.read()))
            for page in pdf.pages:
                text += page.extract_text() or ""
        else:
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")

    # 📝 TEXT INPUT
    elif resume_text:
        text = resume_text

    else:
        return render_index(error="⚠ Please upload a file or enter resume text")

    # 🧠 SKILLS
    skills = extract_skills(text) or []

    # 🎯 RECOMMEND
    recommendations = recommend_logic(skills)

    return render_index(skills=skills, recommendations=recommendations)