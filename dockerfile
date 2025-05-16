FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model, vectorizer, and app code
COPY spam_classifier_nn.joblib .
COPY input_vectorizer.joblib .
COPY app.py .

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]