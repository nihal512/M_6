# train_and_save.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LinearSVC()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate accuracy
acc = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {acc*100:.2f}%")

# app.py (Streamlit app)

import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📧 Email/SMS Spam Classifier")

user_input = st.text_area("Enter your email or SMS message here:")

if st.button("Predict"):
    if user_input.strip():
        # Vectorize input and predict
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        label = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
        st.success(f"Prediction: **{label}**")
    else:
        st.warning("Please enter some text to classify.")
