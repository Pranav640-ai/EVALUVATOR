import requests

data = {
    "text": "FlameSense detects wildfires using satellite thermal imaging to reduce response times."
}

res = requests.post("http://127.0.0.1:5000/generate-quiz", json=data)
print(res.json())
