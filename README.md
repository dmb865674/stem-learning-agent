# ⚡ Low-Bandwidth Multimodal STEM & IT Learning Agent

A lightweight, multimodal educational web application engineered for low-spec mobile devices and low-bandwidth (3G/Edge) environments, powered by **Gemini 2.5 Flash** via the `google-genai` SDK and ready for one-click **Google Cloud Run** deployment.

---

## 🚀 Key Highlights & Architectural Features

### 1. 3-Stage Agentic Educational Workflow
- **Stage 1: Perception & Diagnosis**: Ingests multimodal inputs (handwritten math formulas, code snippets, circuit diagrams, voice recordings, or text). Accurately transcribes the problem, diagnoses the student's cognitive state, and isolates the root misconception or bug.
- **Stage 2: Verification / Code Logic Check**: Executes a rigorous step-by-step mathematical proof or line-by-line dry run of algorithms/circuits (KVL/KCL, recursion depth, boundary conditions, sign changes).
- **Stage 3: Localized Pedagogy & Code-Switching**: Delivers an empathetic, culturally relatable explanation anchored in localized analogies (e.g., Mamak stall order queue for FIFO, Pasar Malam bargaining for binary search, TNB electrical meter for Ohm's law, MRT interchanges for graph nodes) across **Bahasa Melayu**, **Mandarin Chinese**, **Tamil**, **English**, and **Rojak / Manglish**.

### 2. Low-Bandwidth & 3G Optimizations
- **Client-Side Image Compression (< 50 KB)**: Automatically downscales and compresses camera captures and uploaded images in the browser using HTML5 `<canvas>` before network transmission.
- **Live Bandwidth Meter Badge**: Displays original vs. compressed file size in KB and percentage bandwidth saved (typically 95%+ data savings).
- **Low-Bandwidth (Text-Only) Mode**: Disables diagram scripts and minimizes token payloads for remote areas with 2G/3G connectivity.
- **Microphone / Voice Input**: Supports Web Speech API for instantaneous on-device transcription alongside compressed audio note recording.
- **Lightweight Visuals**: Employs on-demand Mermaid.js and SVG diagrams instead of heavy video or high-res raster graphics.
- **Offline Cache**: Automatically saves past lessons in browser `localStorage` with offline network detection (`navigator.onLine`) so students can study without an active internet connection.

---

## 📂 Project Structure

```
stem-learning-agent/
├── main.py              # FastAPI backend with 3-stage agentic workflow & google-genai SDK
├── static/
│   └── index.html       # Ultra-lightweight vanilla HTML5/CSS/JS frontend
├── requirements.txt     # Minimal production Python dependencies
├── Dockerfile           # Production container for Google Cloud Run ($PORT)
├── .dockerignore        # Build optimization rules
└── README.md            # Architecture & deployment documentation
```

---

## 🛠️ Local Development & Quickstart

### Prerequisites
- Python 3.10+
- (Optional) `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/)

### 1. Install Dependencies
```bash
cd stem-learning-agent
pip install -r requirements.txt
```

### 2. Set Environment Variable & Run
```bash
# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key"
python main.py

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-gemini-api-key"
python main.py
```

### 3. Open Web App
Navigate to [http://localhost:8080](http://localhost:8080) in any mobile or desktop browser.

*(Note: If no API key is provided, the app automatically runs in rich Interactive Demo Mode with pre-configured STEM problem scenarios!)*

---

## ☁️ Deploying to Google Cloud Run

### Option A: Direct Build & Deploy with gcloud CLI
```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Deploy directly from source to Cloud Run
gcloud run deploy stem-learning-agent \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY="your-gemini-api-key"
```

### Option B: Build and Push Docker Container
```bash
# 1. Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/stem-learning-agent:latest .

# 2. Push to Google Container Registry / Artifact Registry
docker push gcr.io/YOUR_PROJECT_ID/stem-learning-agent:latest

# 3. Deploy to Cloud Run
gcloud run deploy stem-learning-agent \
  --image gcr.io/YOUR_PROJECT_ID/stem-learning-agent:latest \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars GEMINI_API_KEY="your-gemini-api-key"
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves single-file vanilla frontend |
| `GET` | `/api/health` | Health status and feature check |
| `GET` | `/api/samples` | Quick-start STEM problem templates |
| `POST` | `/api/analyze` | 3-stage multimodal reasoning pipeline (Form/Multipart: `query`, `image`, `audio`, `language`, `persona`, `low_bandwidth`) |
