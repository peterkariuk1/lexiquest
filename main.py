from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv


app = FastAPI(title="Gamified AI Learning Backend - Hackathon Version")
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Gamified AI Learning Backend is running 🚀",
        "url": "http://localhost:8000"
    }

# ---------------------------------------------------------
# 📌 INPUT MODEL
# ---------------------------------------------------------
class LessonRequest(BaseModel):
    prompt: str

# ---------------------------------------------------------
# 📌 SYSTEM PROMPT (Your provided instructions)
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an AI Teaching Assistant specialized in creating lessons for learners with dyslexia.

Task:
- Create a lesson divided into exactly 3 checkpoints.
- Each checkpoint must include:
  1. A short, simple, dyslexia-friendly explanation (2–4 sentences, small words, short sentences).
  2. One quiz question per checkpoint (MCQ, True/False, or drag-and-drop) with correct answer.

Output format (valid JSON only):

{
  "lesson": {
    "checkpoint1": "<checkpoint1 explanation>",
    "quiz1": "<question and correct answer>",
    "checkpoint2": "<checkpoint2 explanation>",
    "quiz2": "<question and correct answer>",
    "checkpoint3": "<checkpoint3 explanation>",
    "quiz3": "<question and correct answer>"
  }
}

Rules:
- Use short sentences.
- Avoid long blocks of text.
- Be warm and friendly, like a teaching avatar.
- No text outside the JSON.
"""

# ---------------------------------------------------------
# 📌 GEMINI MODEL (LangChain)
# ---------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_output_tokens=1024,
    google_api_key=GOOGLE_API_KEY
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_prompt}")
])

# ---------------------------------------------------------
# 📌 GENERATE LESSON ENDPOINT
# ---------------------------------------------------------
@app.post("/generate-lesson")
async def generate_lesson(request: LessonRequest):
    chain = prompt_template | model

    response = chain.invoke({
        "user_prompt": request.prompt
    })

    raw = response.content

    try:
        lesson_json = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        lesson_json = json.loads(match.group(0)) if match else {"error": "Invalid JSON returned"}

    return lesson_json


@app.post("/get-lesson-json")
async def get_lesson_json(request: LessonRequest):
    chain = prompt_template | model

    response = chain.invoke({
        "user_prompt": request.prompt
    })

    raw = response.content

    try:
        lesson_json = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        lesson_json = json.loads(match.group(0)) if match else {"error": "Invalid JSON returned"}

    return {
        "lesson_json": lesson_json
    }
