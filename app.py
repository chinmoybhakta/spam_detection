from flask import Flask, request, jsonify
import joblib
import re

# Load models
clf = joblib.load('F:/chini/flask/models/spam_detector.pkl')
vectorizer = joblib.load('F:/chini/flask/models/vectorizer(en+bn).pkl')

def preprocess_multilingual(text):
    text = text.lower().strip()
    
    # Keep English + Bangla letters
    text = re.sub(r'[^a-z\u0980-\u09FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

app = Flask(__name__)

# Prediction function
def predict_spam(text, threshold=0.55):
    cleaned = preprocess_multilingual(text)
    vect = vectorizer.transform([cleaned])
    prob = clf.predict_proba(vect)[0][1]  # spam probability
    label = "Spam" if prob >= threshold else "Ham"
    return {
        "label": label,
        "spam_probability": round(prob*100, 2)
    }

# Flask route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide 'text' in JSON body"}), 400

    text = data['text']
    result = predict_spam(text)
    return jsonify(result)

# Home page
@app.route('/')
def home():
    return "Multilingual Spam Detector is running!"

if __name__ == '__main__':
    app.run(debug=True)
