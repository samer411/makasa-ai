FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_gemini.py .

CMD ["sh", "-c", "uvicorn api_gemini:app --host 0.0.0.0 --port ${PORT:-8000}"]
