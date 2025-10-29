import requests

payload = {
    "input_content": "Our AI model reduces hospital triage time by 40% using adaptive reinforcement learning.",
    "extra_pillars": [
        {"Pillar_Title": "Deployment Scalability", "Critique": "Can it scale across multiple hospitals?"},
        {"Pillar_Title": "Ethical AI Use", "Critique": "Ensure fairness in patient prioritization."}
    ]
}

res = requests.post("https://evaluvator.onrender.com/evaluate", json=payload)
print(res.json())
