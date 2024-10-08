from flask import Flask, request, jsonify, render_template
from model import generate_idea

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML template

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    generated_idea = generate_idea(prompt)
    return jsonify({"idea": generated_idea})

if __name__ == '__main__':
    app.run(debug=True)
