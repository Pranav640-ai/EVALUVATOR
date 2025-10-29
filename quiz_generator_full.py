import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_quiz_from_text(text):
    """
    Takes input text and generates quiz questions.
    Returns a list of dicts with {question, options, correct_answer}.
    """

    if not text or len(text.strip()) < 30:
        return {"error": "Text too short for quiz generation."}

    prompt = f"""
    You are an AI quiz creator.
    Create 5 high-quality multiple-choice questions based only on this text:
    ---
    {text}
    ---
    Do NOT include names of hackathons, people, or locations.
    Return the result as a JSON array like this:
    [
        {{
            "question": "What is the main purpose of FlameSense?",
            "options": ["Detect wildfires", "Track satellites", "Monitor oceans", "Measure humidity"],
            "correct_answer": "Detect wildfires"
        }}
    ]
    """

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()

    try:
        quiz = json.loads(content)
        return quiz
    except Exception:
        return {"error": "Failed to parse AI response", "raw": content}
