import os
import json
import time
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stem-agent")

app = FastAPI(
    title="Lightweight Multimodal STEM & IT Learning Agent",
    description="Low-bandwidth 3-stage agentic workflow for STEM/IT education powered by Gemini 2.5 Flash",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported language codes & instructions
LANGUAGE_CONFIG = {
    "en": {
        "name": "English",
        "instruction": "Respond in clear, accessible English with relatable everyday analogies (e.g., queues, recipes, traffic, plumbing)."
    },
    "ms": {
        "name": "Bahasa Melayu",
        "instruction": "Balas dalam Bahasa Melayu yang mesra dan mudah difahami, gunakan analogi tempatan (contoh: gerai mamak, pasar malam, kuih lapis, lebuhraya PLUS)."
    },
    "zh": {
        "name": "Mandarin Chinese (华语)",
        "instruction": "使用地道生动的中文解答，搭配生动生活化的比喻（例如：珍珠奶茶配方、排队点餐、红绿灯车流、水管水流）。"
    },
    "ta": {
        "name": "Tamil (தமிழ்)",
        "instruction": "தெளிவான மற்றும் எளிய தமிழில் பதிலளிக்கவும், தினசரி வாழ்க்கை உதாரணங்கள் மற்றும் உவமைகளைப் பயன்படுத்தவும்."
    },
    "rojak": {
        "name": "Rojak / Manglish (Code-Switching)",
        "instruction": "Use natural Southeast Asian code-switching / Manglish / Rojak style (blending English with common conversational BM and local phrases like 'Jom check', 'faham tak', 'tengok sini lah', 'steady') while keeping technical STEM terms precise and crystal clear."
    }
}

PERSONA_CONFIG = {
    "friendly_mentor": "Warm, encouraging, patient STEM mentor who celebrates effort and breaks down complex jargon.",
    "cikgu_santai": "Relaxed, witty local teacher who loves relatable everyday analogies and keeps stress low.",
    "exam_coach": "Laser-focused on marking scheme, common trap questions, step-by-step working marks, and verification.",
    "practical_engineer": "Hands-on, systems-oriented engineer connecting theory to practical real-world devices, circuits, and software."
}

def get_gemini_client(client_key: Optional[str] = None) -> Optional[genai.Client]:
    """Retrieve Gemini client using header key or environment variable."""
    api_key = client_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Error creating genai.Client: {e}")
        return None

def build_agentic_prompt(
    language: str,
    persona: str,
    low_bandwidth: bool,
    has_image: bool,
    has_audio: bool
) -> str:
    lang_info = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["en"])
    persona_desc = PERSONA_CONFIG.get(persona, PERSONA_CONFIG["friendly_mentor"])

    diagram_directive = (
        "Do NOT include complex diagrams; output only raw, compact text/markdown to save bandwidth."
        if low_bandwidth else
        "Include a concise Mermaid.js diagram (graph TD, sequenceDiagram, or flowchart) OR a compact SVG string inside the 'diagram' field if visual clarity aids the concept (e.g. flowchart of logic, circuit block diagram, or state machine). If not applicable, set diagram to null."
    )

    prompt = f"""
You are the **Antigravity Multimodal STEM & IT Learning Agent**, an elite educational AI operating in bandwidth-constrained regions.
Your pedagogical persona: {persona_desc}
Language mode: {lang_info['instruction']}

You must process the user's input through a rigorous 3-Stage Agentic Workflow:

### STAGE 1: Perception & Diagnosis
- Deeply inspect the multimodal input ({'including uploaded photo/diagram' if has_image else ''} {'including recorded student voice' if has_audio else ''}).
- Accurately transcribe and reconstruct any handwritten math formulas, code snippets, logic tables, or circuit schematics.
- Diagnose the user's cognitive state: What is the core question? What was the student attempting? Where is the root misconception, bug, or calculation hurdle?

### STAGE 2: Verification / Code Logic Check
- Perform rigorous step-by-step verification or dry-run execution:
  * For Code / IT: Trace variables line-by-line, test boundary cases (off-by-one, null checks, recursion depth, time complexity).
  * For Math / Physics: Verify every algebraic step, sign change, units of measurement, and physics laws (e.g., KCL/KVL, Newton's laws).
  * For Electronics: Check component polarities, Ohm's law, node voltages, current paths.
- Conclude decisively whether the student's work was correct, partially correct, or flawed.

### STAGE 3: Localized Pedagogy & Analogy
- Deliver a clear, empathetic explanation tailored to the requested language and persona.
- Anchor the solution in a localized, vivid real-world analogy (e.g., Mamak stall order queue for FIFO, Pasar Malam bargaining for binary search, MRT transit interchange for graph nodes, TNB electrical meter for Ohm's law).
- Provide 2-3 concise "Takeaway Rules of Thumb" and 1 short "Quick Check Challenge" question to cement their learning.

Output Format Requirement:
You MUST respond with a valid, clean JSON object matching this exact schema:
{{
  "stage1_diagnosis": {{
    "domain": "Mathematics | Computer Science | Physics | Electronics | General STEM",
    "transcription": "Clean reconstruction of the user formula, code, or query",
    "student_attempt_analysis": "What the user was attempting and where the gap is",
    "core_misconception": "Primary misconception or syntax/logical error"
  }},
  "stage2_verification": {{
    "is_correct": true or false,
    "step_by_step_proof_or_trace": [
      "Step 1: ...",
      "Step 2: ...",
      "Step 3: ..."
    ],
    "verification_verdict": "Detailed explanation of whether the solution works and why"
  }},
  "stage3_pedagogy": {{
    "localized_analogy": "Memorable real-world analogy explaining the core concept",
    "core_explanation_markdown": "Full pedagogical explanation formatted in clean Markdown",
    "rules_of_thumb": [
      "Rule 1...",
      "Rule 2..."
    ],
    "quick_challenge": "A 1-question mini challenge for the student to test their understanding"
  }},
  "diagram": {{
    "type": "mermaid" | "svg" | "none",
    "code": "Mermaid diagram code or inline SVG string, or null if low-bandwidth mode is active"
  }}
}}

{diagram_directive}
Return ONLY valid JSON. Do not wrap in markdown code blocks like ```json ... ```. Just the JSON object.
"""
    return prompt

def generate_mock_response(query: str, language: str, low_bandwidth: bool) -> dict:
    """Fallback interactive demo response when no GEMINI_API_KEY is configured."""
    lang_info = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["en"])
    
    if "circuit" in query.lower() or "resistor" in query.lower() or "ohm" in query.lower():
        return {
            "stage1_diagnosis": {
                "domain": "Electronics & Physics",
                "transcription": query or "Circuit analysis: V = 12V, R1 = 4Ω, R2 = 6Ω in series.",
                "student_attempt_analysis": "Student is calculating total current and voltage drop across each series resistor.",
                "core_misconception": "Often students confuse series vs parallel current rules or forget that current remains constant in a single loop."
            },
            "stage2_verification": {
                "is_correct": False,
                "step_by_step_proof_or_trace": [
                    "Step 1: Total resistance in series R_total = R1 + R2 = 4Ω + 6Ω = 10Ω.",
                    "Step 2: By Ohm's Law (I = V / R_total), total current I = 12V / 10Ω = 1.2A.",
                    "Step 3: Voltage drop V1 = I * R1 = 1.2A * 4Ω = 4.8V.",
                    "Step 4: Voltage drop V2 = I * R2 = 1.2A * 6Ω = 7.2V.",
                    "Step 5: Verify KVL: V1 + V2 = 4.8V + 7.2V = 12.0V (Matches source voltage!)."
                ],
                "verification_verdict": "Verified with Kirchhoff's Voltage Law (KVL). The total current is 1.2A and individual drops sum to 12V."
            },
            "stage3_pedagogy": {
                "localized_analogy": "Analogi Paip Air Taman / TNB: Bayangkan paip air dengan dua penyumbat berturut-turut. Air yang mengalir (Current) mesti sama laju melalui kedua-duanya. Tekanan air (Voltage) yang hilang terbahagi mengikut saiz penyumbat!",
                "core_explanation_markdown": f"### Understanding Series Resistors\n\nIn a single loop circuit, electrical charge has only **one single road** to travel. Therefore:\n- **Current ($I$) is identical** through every series element: $I_1 = I_2 = I_{{total}} = 1.2\\text{{ A}}$.\n- **Voltage drops** proportionally to resistance according to $V = I \\times R$.\n\n*{lang_info['instruction']}*",
                "rules_of_thumb": [
                    "Series Circuit: Current is constant, Voltages add up.",
                    "Parallel Circuit: Voltage is constant, Currents add up.",
                    "Always check: The sum of voltage drops MUST equal the supply voltage (KVL)."
                ],
                "quick_challenge": "If we swap R2 with a 2Ω resistor, what will the new total current be from the 12V supply?"
            },
            "diagram": {
                "type": "none" if low_bandwidth else "mermaid",
                "code": "" if low_bandwidth else "graph LR\n  Source[12V DC Supply] -->|1.2A| R1[Resistor 1: 4Ω / Drop: 4.8V]\n  R1 -->|1.2A| R2[Resistor 2: 6Ω / Drop: 7.2V]\n  R2 -->|Ground| Source"
            }
        }
    else:
        # Default CS / Algorithm Demo
        return {
            "stage1_diagnosis": {
                "domain": "Computer Science & Algorithms",
                "transcription": query or "def search(arr, target): return binary_search(arr, target)",
                "student_attempt_analysis": "Student is implementing Binary Search or analyzing list lookup complexity.",
                "core_misconception": "Forgetting that Binary Search strictly requires a pre-sorted array, or encountering integer overflow on midpoint calculation (low + high) // 2."
            },
            "stage2_verification": {
                "is_correct": True,
                "step_by_step_proof_or_trace": [
                    "Step 1: Check preconditions: array must be sorted in ascending order.",
                    "Step 2: Midpoint formula: mid = low + (high - low) // 2 prevents potential 32-bit integer overflow.",
                    "Step 3: Compare arr[mid] with target: if equal, return mid.",
                    "Step 4: If target < arr[mid], adjust high = mid - 1. Else low = mid + 1.",
                    "Step 5: Loop terminates when low > high, returning -1 for not found. Time Complexity: O(log N)."
                ],
                "verification_verdict": "Algorithm logic verified. Halves search space each iteration, making it exponentially faster than linear O(N) search."
            },
            "stage3_pedagogy": {
                "localized_analogy": "Analogi Tawar Menawar Pasar Malam / Kamus Tebal: Bila cari perkataan dalam kamus 1000 muka surat, kita tak selak satu-satu dari page 1. Kita buka tengah-tengah! Kalau huruf yang dicari lebih depan, kita buang separuh belakang serta-merta. Jimat masa!",
                "core_explanation_markdown": f"### Binary Search Masterclass\n\nBinary Search is the epitome of the **Divide and Conquer** paradigm.\n\n```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = low + (high - low) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```\n\n*{lang_info['instruction']}*",
                "rules_of_thumb": [
                    "Array MUST be sorted before Binary Search works.",
                    "Each comparison eliminates 50% of remaining items: $\\log_2(1,000,000) \\approx 20$ steps only!",
                    "Boundary condition: Use `low <= high` with `mid - 1` and `mid + 1`."
                ],
                "quick_challenge": "In a sorted array with 1,024 elements, what is the maximum number of comparisons Binary Search needs in the worst case?"
            },
            "diagram": {
                "type": "none" if low_bandwidth else "mermaid",
                "code": "" if low_bandwidth else "graph TD\n  Start[Array of 1000 items] --> Mid{Check Midpoint}\n  Mid -->|Target Smaller| Left[Search Left Half: 500 items]\n  Mid -->|Target Greater| Right[Search Right Half: 500 items]\n  Mid -->|Match!| Found[Return Index]\n  Left --> Repeat[Repeat halving... Max 10 steps!]"
            }
        }

@app.get("/api/health")
async def health_check():
    has_env_key = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "healthy",
        "service": "stem-learning-agent",
        "model": "gemini-2.5-flash",
        "gemini_api_key_configured": has_env_key,
        "features": [
            "3-Stage Agentic Workflow (Perception, Verification, Localized Pedagogy)",
            "Multimodal: Photo OCR, Audio Voice, Text Query",
            "Client-side HTML5 Canvas Image Compression (<50KB)",
            "Low-Bandwidth Text-Only Mode",
            "Multilingual Support (BM, Mandarin, Tamil, English, Rojak)",
            "Offline LocalStorage Cache"
        ]
    }

@app.get("/api/samples")
async def get_samples():
    """Returns ready-to-test multimodal STEM sample problems."""
    return [
        {
            "id": "py_recursion",
            "title": "Python: Infinite Recursion Bug",
            "category": "Computer Science",
            "query": "Why does my recursive function crash with RecursionError: maximum recursion depth exceeded?\n\ndef countdown(n):\n    print(n)\n    return countdown(n - 1)",
            "language": "rojak",
            "persona": "friendly_mentor"
        },
        {
            "id": "physics_circuits",
            "title": "Electronics: Series-Parallel Resistors",
            "category": "Electronics & Physics",
            "query": "A 12V battery powers two resistors R1=4 ohms and R2=6 ohms connected in series. Find the total current and the voltage drop across each resistor.",
            "language": "ms",
            "persona": "cikgu_santai"
        },
        {
            "id": "math_calculus",
            "title": "Calculus: Derivative of Velocity",
            "category": "Mathematics",
            "query": "Given position function s(t) = 3t^2 - 12t + 9, find the instantaneous velocity when t = 4s, and find when the object is momentarily at rest.",
            "language": "en",
            "persona": "exam_coach"
        },
        {
            "id": "cs_bubble_sort",
            "title": "Algorithms: Bubble Sort Optimization",
            "category": "Computer Science",
            "query": "How does adding a 'swapped' boolean flag optimize Bubble Sort from O(n^2) to O(n) for already-sorted arrays?",
            "language": "zh",
            "persona": "practical_engineer"
        }
    ]

@app.post("/api/analyze")
async def analyze_stem_problem(
    request: Request,
    query: Optional[str] = Form(None),
    language: str = Form("en"),
    persona: str = Form("friendly_mentor"),
    low_bandwidth: bool = Form(False),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    x_gemini_api_key: Optional[str] = Header(None)
):
    start_time = time.time()
    query_text = (query or "").strip()
    
    # Process image if provided
    image_bytes = None
    image_mime = None
    original_image_size = 0
    if image and image.filename:
        image_bytes = await image.read()
        original_image_size = len(image_bytes)
        image_mime = image.content_type or "image/jpeg"
        logger.info(f"Received image: {image.filename}, size: {original_image_size} bytes, mime: {image_mime}")

    # Process audio if provided
    audio_bytes = None
    audio_mime = None
    original_audio_size = 0
    if audio and audio.filename:
        audio_bytes = await audio.read()
        original_audio_size = len(audio_bytes)
        audio_mime = audio.content_type or "audio/webm"
        logger.info(f"Received audio: {audio.filename}, size: {original_audio_size} bytes, mime: {audio_mime}")

    if not query_text and not image_bytes and not audio_bytes:
        raise HTTPException(status_code=400, detail="Please provide a text question, an uploaded photo, or a voice recording.")

    client = get_gemini_client(x_gemini_api_key)
    
    # If no Gemini API key is configured anywhere, provide high quality simulated pedagogical response
    if not client:
        logger.warning("No GEMINI_API_KEY found; returning rich pedagogical demo response.")
        demo_result = generate_mock_response(query_text, language, low_bandwidth)
        latency = round((time.time() - start_time) * 1000, 2)
        response_bytes = len(json.dumps(demo_result).encode('utf-8'))
        
        return JSONResponse(content={
            **demo_result,
            "meta": {
                "mode": "demo_fallback",
                "message": "Demo mode (Set GEMINI_API_KEY in environment or UI settings for live Gemini 2.5 Flash)",
                "latency_ms": latency,
                "response_size_kb": round(response_bytes / 1024, 2),
                "image_received_kb": round(original_image_size / 1024, 2) if original_image_size else 0,
                "audio_received_kb": round(original_audio_size / 1024, 2) if original_audio_size else 0,
                "low_bandwidth_active": low_bandwidth
            }
        })

    # Prepare multimodal contents for Gemini 2.5 Flash
    prompt = build_agentic_prompt(
        language=language,
        persona=persona,
        low_bandwidth=low_bandwidth,
        has_image=bool(image_bytes),
        has_audio=bool(audio_bytes)
    )

    contents = []
    
    # Append image part if exists
    if image_bytes:
        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
            contents.append(image_part)
        except Exception as e:
            logger.error(f"Failed to create image Part: {e}")

    # Append audio part if exists
    if audio_bytes:
        try:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime)
            contents.append(audio_part)
        except Exception as e:
            logger.error(f"Failed to create audio Part: {e}")

    # Append user query + agentic system instructions
    user_message = f"User STEM Problem / Query:\n{query_text if query_text else '(Analyze the provided image/audio)'}\n\n{prompt}"
    contents.append(user_message)

    try:
        gen_config = types.GenerateContentConfig(
            temperature=0.2 if low_bandwidth else 0.4,
            response_mime_type="application/json"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=gen_config
        )

        raw_text = response.text.strip()
        # Clean potential markdown wrapping if returned despite prompt
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        parsed_response = json.loads(raw_text)
        latency = round((time.time() - start_time) * 1000, 2)
        response_bytes = len(raw_text.encode('utf-8'))

        return JSONResponse(content={
            **parsed_response,
            "meta": {
                "mode": "gemini-2.5-flash",
                "latency_ms": latency,
                "response_size_kb": round(response_bytes / 1024, 2),
                "image_received_kb": round(original_image_size / 1024, 2) if original_image_size else 0,
                "audio_received_kb": round(original_audio_size / 1024, 2) if original_audio_size else 0,
                "low_bandwidth_active": low_bandwidth
            }
        })

    except Exception as e:
        logger.exception(f"Gemini generation error: {e}")
        # Provide graceful error with mock fallback to avoid student disruption
        demo_result = generate_mock_response(query_text, language, low_bandwidth)
        latency = round((time.time() - start_time) * 1000, 2)
        return JSONResponse(content={
            **demo_result,
            "meta": {
                "mode": "error_fallback",
                "error": str(e),
                "message": f"Encountered API issue: {str(e)[:100]}. Rendered robust fallback response.",
                "latency_ms": latency,
                "response_size_kb": 2.1,
                "low_bandwidth_active": low_bandwidth
            }
        })

# Mount static files directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
