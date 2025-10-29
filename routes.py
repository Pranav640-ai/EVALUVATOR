from flask import Flask, request, jsonify
from flask_cors import CORS
from evaluvator import run_dynamic_evaluation

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

@app.route('/')
def home():
    return jsonify({
        "message": "🔥 Presentation Evaluator API is running",
        "endpoints": {
            "POST /evaluate": "Evaluate presentation content with optional extra pillars"
        }
    })

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    POST endpoint to evaluate presentation content.
    Expected JSON structure:
    {
        "input_content": "Your presentation text here",
        "extra_pillars": [
            {"Pillar_Title": "New Pillar 1", "Critique": "Your critique"},
            {"Pillar_Title": "New Pillar 2", "Critique": "Another one"}
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or "input_content" not in data:
            return jsonify({"error": "Missing required field 'input_content'"}), 400

        input_content = data["input_content"]
        extra_pillars = data.get("extra_pillars", [])

        result = run_dynamic_evaluation(input_content, extra_pillars)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
