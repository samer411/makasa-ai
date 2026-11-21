import base64
import json
import os

import pymupdf                   # PyMuPDF (only needed if PDF_DIRECT is off)
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Configuration (environment variables) ----------------
API_KEY = os.getenv("GEMINI_API_KEY")
# Do not crash at startup: /health will report whether the key is configured.

# Try these stable, free-tier models in order. Avoid preview/exp models.
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-2.5-flash,gemini-2.0-flash"
).split(",") if m.strip()]

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Comma-separated list of exact form origins, e.g. http://forms.nile.edu,https://forms.nile.edu
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "LF_ORIGIN", "http://your-laserfiche-server"
).split(",") if o.strip()]
MAX_PAGES = int(os.getenv("MAX_PAGES", "4"))
ZOOM = float(os.getenv("PDF_ZOOM", "1.5"))
# Send original PDF bytes straight to Gemini (better OCR, bigger request).
# Set to "1" if your scans are not too large. Otherwise we rasterize pages.
PDF_DIRECT = os.getenv("GEMINI_PDF_DIRECT", "0") == "1"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Same bilingual prompt as the local version ----------------
PROMPT = """You read scanned university transcript documents written in BOTH Arabic and English.

The document contains TWO tables:
1. Courses passed at the transferring university —
   "Courses Passed by the Student at the Transferring University" /
   "مقررات درسها الطالب بنجاح في الجامعة المحول منها"
2. Equivalent / exempted courses at the receiving university —
   "Equivalent Courses Exempted for the Student at the Receiving University" /
   "مقررات مناظرة أعفي منها الطالب في الجامعة المحول إليها"

Return ONLY a JSON object with exactly two keys: "passed" and "exempted".
Each key holds an array of objects with exactly these keys:
code, name, grade, hours, desc

BILINGUAL RULES (very important):
- Read BOTH Arabic and English text, including cells that mix both languages.
- Copy Arabic exactly in Arabic script. Do NOT translate and do NOT transliterate it.
- Copy English exactly as written. Do NOT translate it.
- If a cell contains both languages, keep both in the same string.
- Course codes and numbers are language-neutral — copy them exactly.
- Unreadable or missing value -> "" ; absent section -> [].

Field mapping:
code  = Course Code / الرقم الكودي للمقرر
name  = Course Name / إسم المقرر
grade = Grade / التقدير
hours = Credit Hours or Semester / الساعات أو الفصل الدراسي
desc  = Brief Course Description / وصف مختصر للمقرر

No markdown, no explanation — only JSON."""


# ---------------- Helpers ----------------
def render_pdf(data: bytes):
    """Rasterize the first MAX_PAGES pages of a PDF into PNG bytes."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    return [
        p.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM)).tobytes("png")
        for p in list(doc)[:MAX_PAGES]
    ]


def parse_model_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def ask_gemini(parts):
    """Try each configured Gemini model until one returns valid JSON."""
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set on the server. "
            "Add it in Railway Variables and redeploy."
        )
    last_error = None
    for model in MODELS:
        try:
            r = requests.post(
                BASE_URL.format(model=model),
                params={"key": API_KEY},
                json={
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0,
                    },
                },
                timeout=300,
            )
            if not r.ok:
                raise RuntimeError("HTTP %s: %s" % (r.status_code, r.text[:500]))
            body = r.json()
            if "candidates" not in body or not body["candidates"]:
                raise RuntimeError("No candidates returned: %s" % json.dumps(body)[:500])
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            return parse_model_json(text)
        except Exception as e:
            last_error = e
            print("[warn] model %s failed: %s" % (model, e))
    raise last_error


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "provider": "gemini",
        "models": MODELS,
        "key_configured": bool(API_KEY),
    }


@app.post("/api/extract")
async def extract(files: list[UploadFile] = File(...)):
    parts = [{"text": PROMPT}]

    for f in files:
        raw = await f.read()
        name = (f.filename or "").lower()
        if name.endswith(".pdf"):
            if PDF_DIRECT:
                parts.append({
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": base64.b64encode(raw).decode(),
                    }
                })
            else:
                for img in render_pdf(raw):
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(img).decode(),
                        }
                    })
        else:
            parts.append({
                "inline_data": {
                    "mime_type": f.content_type or "image/png",
                    "data": base64.b64encode(raw).decode(),
                }
            })

    out = ask_gemini(parts)
    return {
        "passed": out.get("passed", []),
        "exempted": out.get("exempted", []),
    }
