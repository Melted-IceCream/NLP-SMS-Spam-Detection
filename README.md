This is a simple neural network to detect SMS spam. The dataset can be found on [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

# Results
> Accuracy: 0.981166
>
> F1: 0.926829
>
> Precision: 0.970803
>
> Recall: 0.886667

# Data Preprocessing
1. Special characters and punctuations are removed from the data as it has no significant value to the neural network.
> word.translate(str.maketrans('', '', string.punctuation))
2. The text is then vectorized so that the neural network can understand it easier. Stopwords removal is also built into the TFIDF vectorizer function.
> from sklearn.feature_extraction.text import TfidfVectorizer
>
> vectorizer = TfidfVectorizer(stop_words='english')
> 
> X_train = vectorizer.fit_transform(X_train)
> 
> X_test = vectorizer.transform(X_test)

# Model Architechture
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

# Deployment
1. Make a directory for the project.
   ```
   mkdir Project
   ```
2. Clone the project.
   ```
   git clone https://github.com/Melted-IceCream/SMS-Spam-Detection.git
   ```
3. Open the docker desktop app.
4. Go into the folder named docker and build an image.
   ```
   cd docker
   ```
   ```
   docker build -t sms-spam-detection .
   ```
5. Run the image.
   ```
   docker run -p 5000:5000 sms-spam-detection
   ```
6. Visit the site http://localhost:5000 or use the link in the docker desktop app to run the application.
