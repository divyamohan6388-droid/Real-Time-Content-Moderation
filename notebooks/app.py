# loading model

import streamlit as st
import joblib
import string

# -------------------------------
# Load Models and TF-IDF Vectorizer
# -------------------------------

tfidf = joblib.load("../data/tfidf_vectorizer.pkl")

lr_model = joblib.load("../data/logistic_regression_model.pkl")

# -------------------------------
# Text Cleaning Function
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = " ".join(text.split())
    return text

# -------------------------------
# Streamlit Page Configuration
# -------------------------------

st.set_page_config(
    page_title="AI Content Moderator",
    page_icon="🛡️",
    layout="centered"
)

# -------------------------------
# Website Title
# -------------------------------

st.title("🛡️ AI Content Moderation System")

st.write(
    "Detect spam, abusive, and clean messages "
    "using Machine Learning."
)

st.divider()

# -------------------------------
# Message Input
# -------------------------------

message = st.text_area(
    "Enter a message to moderate:",
    placeholder="Type your message here...",
    height=150
)
# adding the moderation button and prediction
# -------------------------------
# Moderation Button
# -------------------------------

if st.button("🔍 Moderate Message", type="primary"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Clean the input message
        cleaned_message = clean_text(message)

        # Convert message into TF-IDF features
        message_tfidf = tfidf.transform([cleaned_message])

        # Make prediction
        prediction = str(lr_model.predict(message_tfidf)[0])

        # Get prediction probabilities
        probabilities = lr_model.predict_proba(message_tfidf)[0]

        # Map probabilities to labels
        class_probabilities = dict(
            zip(lr_model.classes_, probabilities)
        )

        # Display result
        st.subheader("Moderation Result")

        st.write(f"**Prediction:** {prediction.upper()}")

        st.write("### Confidence Scores")

        for label, probability in class_probabilities.items():
            st.write(f"**{label.upper()}**: {probability:.2%}")