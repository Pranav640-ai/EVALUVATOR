# presentation_evaluator_v2.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json, os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import json

import os
from dotenv import load_dotenv
from dotenv import load_dotenv
import os

load_dotenv()  # this must come before you read environment variables
print("Loaded API Key:", os.getenv("OPENROUTER_API_KEY"))

# ===================================
# 🧠 1. Default Pillars and Structure
# ===================================
from langdetect import detect
import re

from openai import OpenAI
import os
import os
import requests
from openai import OpenAI

def is_meaningless_text(text: str) -> bool:
    """
    Detects if input text is meaningful (presentation-like) or meaningless.
    Returns True if meaningless, False if valid.
    """

    # Step 1: Quick heuristic check
    if len(text.strip().split()) < 5:
        return True

    # Step 2: Try primary AI (Gemini via OpenRouter)
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        prompt = f"""
        You are a classifier. Determine whether the following text is meaningful presentation content
        (like a real paragraph, topic description, or idea), or just random words, gibberish, or nonsense.

        Text:
        \"\"\"{text}\"\"\"

        Reply with only one word:
        - VALID → meaningful or structured presentation content.
        - INVALID → random, meaningless, or nonsensical.
        """

        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[{"role": "user", "content": prompt}],
        )

        ai_reply = response.choices[0].message.content.strip().upper()

        return "INVALID" in ai_reply

    except Exception as e:
        print(f"⚠️ AI-based check via Gemini failed ({e}), trying backup API...")

    # Step 3: Fallback → use free model from ApiFreeLLM
import os
import re
import requests
from openai import OpenAI

def is_meaningless_text(text: str) -> bool:
    """
    Detects if text is gibberish or meaningless using a three-layer fallback system:
    1. OpenRouter (Gemini) AI classifier
    2. Hugging Face GPT-2 model (public endpoint)
    3. Local heuristic rules (no network)
    
    Returns:
        True  → meaningless text
        False → meaningful presentation content
    """

    text = text.strip()
    if len(text.split()) < 5:
        return True  # Too short to be meaningful

    # =========================
    # 1️⃣ Primary: Gemini via OpenRouter
    # =========================
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        prompt = f"""
        You are a strict text classifier.
        Decide if the following text is meaningful presentation content
        (like an idea, project description, or paragraph),
        or just random/gibberish words.

        Text:
        \"\"\"{text}\"\"\"

        Reply with only one word:
        - "VALID" for meaningful content
        - "INVALID" for nonsense
        """

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            timeout=15,
        )

        ai_reply = response.choices[0].message.content.strip().upper()
        if "INVALID" in ai_reply:
            return True
        elif "VALID" in ai_reply:
            return False
        else:
            raise ValueError("Unexpected AI reply")

    except Exception as e:
        print(f"⚠️ Gemini check failed ({e}), trying HuggingFace backup...")

    # =========================
    # 2️⃣ Backup: Hugging Face GPT-2
    # =========================
    try:
        headers = {}
        if os.getenv("HUGGINGFACE_API_KEY"):
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

        hf_payload = {
            "inputs": f"Classify the text as VALID or INVALID:\n\n{text}\n\nYour answer:"
        }

        hf_response = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers=headers,
            json=hf_payload,
            timeout=15,
        )

        if hf_response.status_code == 200:
            data = hf_response.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                hf_reply = data[0]["generated_text"].upper()
                if "INVALID" in hf_reply:
                    return True
                elif "VALID" in hf_reply:
                    return False
        else:
            print(f"⚠️ HuggingFace API error {hf_response.status_code}")

    except Exception as e:
        print(f"⚠️ HuggingFace backup failed ({e}), using local heuristic...")

    # =========================
    # 3️⃣ Final Fallback: Local Heuristic
    # =========================
    nonsense_patterns = [
        r"(.)\1{3,}",                     # long repeated characters
        r"[bcdfghjklmnpqrstvwxyz]{6,}",   # long consonant strings
        r"[^a-zA-Z0-9\s]{4,}",            # too many symbols
    ]

    if any(re.search(p, text.lower()) for p in nonsense_patterns):
        return True

    # if mostly non-English or random words
    non_alpha_ratio = sum(1 for c in text if not c.isalpha()) / max(len(text), 1)
    if non_alpha_ratio > 0.5:
        return True

    # fallback word length check
    return len(text.split()) < 5



DEFAULT_FIELDS = {
    "AI_Persona": "Expert Strategic Consultant and high-stakes presentation evaluator.",
    "Input_Content": "gytyhtrytrytrytyhfgghgfhrth",
    "Task": (
        "Analyze the Input_Content to identify and critically score the content "
        "against the Eight Pillars of Professional Communication (1-8). For each pillar, "
        "assign a numeric score (1–10) and provide a short critique. Then calculate a final "
        "average score across all pillars. Provide specific, actionable critique referencing "
        "the 'Narrative Hook' and 'Unique Value Proposition (UVP)/Novelty'."
    ),
    "Output_Required_Format": "JSON",
    "Output_Fields": {
        "Overall_Presentation_Rating": {
            "Average_Score": "/10",
            "Score_Percentage": "/100",
            "Justification": "Summary justification based on all pillars."
        },
        "Pillars_of_Evaluation": [
            {
                "Pillar_Number": 1,
                "Pillar_Title": "Core Relevance & Narrative Hook",
                "Score": "/10",
                "Critique": "Does the opening create immediate relevance? Rate emotional impact (1–10)."
            },
            {
                "Pillar_Number": 2,
                "Pillar_Title": "Central Thesis or Problem Definition",
                "Score": "/10",
                "Critique": "Is the main problem clearly defined? Avoid over-technical or vague phrasing."
            },
            {
                "Pillar_Number": 3,
                "Pillar_Title": "Unique Value Proposition (UVP) & Novelty",
                "Score": "/10",
                "Critique": "How strong and novel is the UVP? Suggest differentiation if weak."
            },
            {
                "Pillar_Number": 4,
                "Pillar_Title": "Stakeholder Impact & Broad Application",
                "Score": "/10",
                "Critique": "Are 3+ distinct groups impacted? Assess the scope breadth."
            },
            {
                "Pillar_Number": 5,
                "Pillar_Title": "Methodology and Technical Authority",
                "Score": "/10",
                "Critique": "Does the method show authority, scale, and reliability?"
            },
            {
                "Pillar_Number": 6,
                "Pillar_Title": "Quantifiable Value & ROI",
                "Score": "/10",
                "Critique": "Identify quantifiable metrics (e.g., % gain, time saved)."
            },
            {
                "Pillar_Number": 7,
                "Pillar_Title": "Limitations, Risks, and Mitigation Strategy",
                "Score": "/10",
                "Critique": "Did the presenter acknowledge and address risks realistically?"
            },
            {
                "Pillar_Number": 8,
                "Pillar_Title": "Delivery, Authority, and Credibility",
                "Score": "/10",
                "Critique": "Assess clarity, confidence, and delivery flow."
            }
        ]
    }
}
class Pillar(BaseModel):
    Pillar_Number: int = Field(..., description="The index number of the evaluation pillar.")
    Pillar_Title: str = Field(..., description="The name/title of the evaluation pillar.")
    Score: int = Field(..., description="Numeric score between 1–10.")
    Critique: str = Field(..., description="Short critique or analysis for this pillar.")

class OverallRating(BaseModel):
    Average_Score: float = Field(..., description="Average score out of 10.")
    Score_Percentage: float = Field(..., description="Final score in percentage (0–100).")
    Justification: str = Field(..., description="Brief justification summary.")

class EvaluationResult(BaseModel):
    Overall_Presentation_Rating: OverallRating
    Pillars_of_Evaluation: list[Pillar]

# ====================================
# 🧾 2. Input Validation & Customization
# ====================================
def validate_fields(user_json: dict):
    """Ensure all required fields exist and fill missing defaults."""
    missing = [key for key in DEFAULT_FIELDS if key not in user_json]
    extras = [key for key in user_json if key not in DEFAULT_FIELDS]

    if missing:
        print(f"⚠️ Missing default fields added: {missing}")
    if extras:
        print(f"ℹ️ Extra fields detected: {extras}")

    for key in missing:
        user_json[key] = DEFAULT_FIELDS[key]
    return user_json


def add_custom_fields(user_json: dict, extra_pillars: list[dict] | None = None):
    """
    Dynamically merges extra pillars passed from backend.
    If extra_pillars is None or empty, default pillars remain unchanged.
    """
    if extra_pillars:
        existing = user_json["Output_Fields"]["Pillars_of_Evaluation"]
        start_index = len(existing) + 1

        for i, pillar in enumerate(extra_pillars, start=start_index):
            new_field = {
                "Pillar_Number": i,
                "Pillar_Title": pillar.get("Pillar_Title", f"Custom Pillar {i}"),
                "Score": "/10",
                "Critique": pillar.get("Critique", "No critique provided.")
            }
            existing.append(new_field)
            print(f"✅ Added new backend pillar: {new_field['Pillar_Title']}")

    return user_json



# ==================================
# 💬 3. LangChain Prompt Definition
# ==================================
EVALUATION_PROMPT = ChatPromptTemplate.from_template("""
You are {AI_Persona}.

Task:
{Task}

Analyze and score this content:
{Input_Content}

For each evaluation pillar, assign a numeric score from 1 to 10 and a brief critique.
Then compute the final average score and percentage.

Return ONLY valid JSON structured exactly as follows:
{Output_Fields}
""")


# ====================================
# 🤖 4. Core Evaluation Function (LLM)
# ====================================
def evaluate_presentation(input_json, extra_pillars=None):

    validated = validate_fields(input_json)
    customized = add_custom_fields(validated, extra_pillars)


    if is_meaningless_text(customized["Input_Content"]):
        print("❌ Input appears meaningless or too short to evaluate.")
        return {"error": "Invalid input: not a meaningful presentation or pitch."}

    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    structured_llm = llm.with_structured_output(EvaluationResult)

    prompt = ChatPromptTemplate.from_template("""
You are {AI_Persona}.

Task:
{Task}

Analyze and score this presentation content:
{Input_Content}

Use this dynamically provided output schema for your JSON structure:
{Output_Fields}

Return ONLY valid JSON strictly following this structure.
""")

    chain = prompt | structured_llm

    # ✅ Debug check
    print("🧩 Active Pillars in Schema:")
    for p in customized["Output_Fields"]["Pillars_of_Evaluation"]:
        print(f"  - {p['Pillar_Number']}. {p['Pillar_Title']}")

    result = chain.invoke({
        "AI_Persona": customized["AI_Persona"],
        "Task": customized["Task"],
        "Input_Content": customized["Input_Content"],
        "Output_Fields": json.dumps(customized["Output_Fields"], indent=2)
    })

    return result.dict()


def run_dynamic_evaluation(input_content: str, extra_pillars: list[dict] | None = None) -> dict:
    """
    Accepts raw presentation text, optionally lets the user add extra evaluation pillars,
    runs the AI evaluation, and returns the structured JSON result.
    """
    # Step 1: Prepare base JSON
    user_json = {
        "AI_Persona": DEFAULT_FIELDS["AI_Persona"],
        "Task": DEFAULT_FIELDS["Task"],
        "Input_Content": input_content.strip(),
        "Output_Fields": DEFAULT_FIELDS["Output_Fields"].copy()
    }

    # Step 2: Validate & allow adding custom pillars
    validated = validate_fields(user_json)
    customized = add_custom_fields(validated, extra_pillars)


    # Step 3: Check if text is meaningful
    if is_meaningless_text(customized["Input_Content"]):
        print("❌ Input appears meaningless or too short to evaluate.")
        return {"error": "Invalid input: not a meaningful presentation or pitch."}

    # Step 4: Initialize model
    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    structured_llm = llm.with_structured_output(EvaluationResult)

    # Step 5: Prepare prompt
    prompt = ChatPromptTemplate.from_template("""
You are {AI_Persona}.

Task:
{Task}

Analyze and score this presentation content:
{Input_Content}

Use this dynamically provided output schema for your JSON structure:
{Output_Fields}

Return ONLY valid JSON strictly following this structure.
""")

    chain = prompt | structured_llm

    # Debug log
    print("\n🧩 Active Evaluation Pillars:")
    for p in customized["Output_Fields"]["Pillars_of_Evaluation"]:
        print(f"  - {p['Pillar_Number']}. {p['Pillar_Title']}")

    # Step 6: Invoke model
    result = chain.invoke({
        "AI_Persona": customized["AI_Persona"],
        "Task": customized["Task"],
        "Input_Content": customized["Input_Content"],
        "Output_Fields": json.dumps(customized["Output_Fields"], indent=2)
    })

    # Step 7: Return structured result
    return result.dict()

# ====================================
# 🧪 5. Example Run
# ====================================
if __name__ == "__main__":
    example_input = "FlameSense uses satellite thermal imaging and IoT edge sensors to detect wildfires within 30 seconds, reducing false alarms by 45% and saving $8.2M in six months."

    extra_pillars = [
        {"Pillar_Title": "Scalability Potential", "Critique": "Evaluate the ability to scale to multiple regions."},
        {"Pillar_Title": "Ethical and Environmental Impact", "Critique": "Assess awareness of sustainability and ethical implications."}
    ]

    result = run_dynamic_evaluation(example_input, extra_pillars)

    print("\n✅ Evaluation Result:")
    print(json.dumps(result, indent=2))

