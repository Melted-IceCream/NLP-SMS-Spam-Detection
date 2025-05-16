from flask import Flask, request, render_template_string
import joblib
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_classifier_nn.joblib')
vectorizer = joblib.load('input_vectorizer.joblib')

# HTML form template
HTML_FORM = """
<!doctype html>
<html>
<head><title>Spam Classifier Test</title></head>
<body>
  <h1>Test Spam Detection</h1>
  <form method="POST">
    <textarea name="text" rows="4" cols="50" placeholder="Enter message..."></textarea><br>
    <button type="submit">Check</button>
  </form>
{% if result %}
    <div style="background: {{ '#ffcccc' if result.prediction == 'spam' else '#ccffcc' }};
                padding: 10px; margin-top: 20px; border-radius: 5px;">
        <strong>Prediction:</strong> {{ result.prediction|upper }}<br>
        <strong>Confidence:</strong> {{ (result.confidence * 100)|round(1) }}%
        <div style="margin-top: 5px; height: 5px; background: #ddd; width: 100%">
            <div style="height: 100%; width: {{ (result.confidence * 100)|round(1) }}%; 
                        background: {{ '#ff0000' if result.prediction == 'spam' else '#00aa00' }};"></div>
        </div>
    </div>
{% endif %}
</body>
</html> """

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']

        #Clean up data by removing punctuation
        def clean(word):
            return word.translate(str.maketrans('', '', string.punctuation))
        
        no_punctuation = clean(text)
        X = vectorizer.transform([no_punctuation]).toarray()
        prediction = model.predict(X)[0][0]  # Get probability
        result = {
            'prediction': 'spam' if prediction > 0.5 else 'ham',
            # Confidence is absolute distance from decision boundary (0.5)
            'confidence': abs(prediction - 0.5) * 2  # Scales to 0-100% range
        }
    return render_template_string(HTML_FORM, result=result)

app.run(host='0.0.0.0', port=5000)