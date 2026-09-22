import streamlit as st
import joblib
import string
import pandas as pd
from pathlib import Path
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Aegis AI | Content Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODELS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

possible_data_dirs = [
    BASE_DIR / "data",
    BASE_DIR.parent / "data",
    Path.cwd() / "data",
    Path.cwd().parent / "data"
]

DATA_DIR = None

for folder in possible_data_dirs:
    vectorizer_path = folder / "tfidf_vectorizer.pkl"
    model_path = folder / "logistic_regression_model.pkl"

    if vectorizer_path.exists() and model_path.exists():
        DATA_DIR = folder
        break

if DATA_DIR is None:
    st.error(
        "Model files were not found. Please check your data folder."
    )
    st.stop()

try:
    tfidf = joblib.load(
        DATA_DIR / "tfidf_vectorizer.pkl"
    )

    lr_model = joblib.load(
        DATA_DIR / "logistic_regression_model.pkl"
    )

except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# TEXT PREPROCESSING
# =====================================================

def clean_text(text):
    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text

# =====================================================
# MODERATION FUNCTION
# =====================================================

def moderate_message(message):

    cleaned_message = clean_text(message)

    message_tfidf = tfidf.transform(
        [cleaned_message]
    )

    prediction = str(
        lr_model.predict(message_tfidf)[0]
    )

    probabilities = lr_model.predict_proba(
        message_tfidf
    )[0]

    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            lr_model.classes_,
            probabilities
        )
    }

    return prediction, class_probabilities

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🛡️ Aegis AI")

    st.caption("Content Intelligence Platform")

    st.divider()

    st.subheader("Project Information")

    st.write(
        "A machine learning application for classifying "
        "messages into clean, spam, and abusive categories."
    )

    st.write("**Machine Learning:** Logistic Regression")

    st.write("**Feature Extraction:** TF-IDF")

    st.write("**Framework:** Streamlit")

    st.write("**Field:** NLP and Content Moderation")

    st.divider()

    st.subheader("Model Categories")

    st.success("🟢 Clean")

    st.warning("🟠 Spam")

    st.error("🔴 Abusive")

    st.divider()

    st.caption("Built by Divya Mohan | IIT Patna")

# =====================================================
# HEADER
# =====================================================

st.title("🛡️ Aegis AI")

st.subheader(
    "Intelligent Content Moderation System"
)

st.write(
    "Analyse messages using Natural Language Processing "
    "and Machine Learning. Detect potential spam and "
    "abusive content with real-time classification."
)

st.success("● Machine Learning System Online")

st.divider()

# =====================================================
# METRICS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Model",
        value="Logistic Regression"
    )

with col2:
    st.metric(
        label="Features",
        value="TF-IDF"
    )

with col3:
    st.metric(
        label="Categories",
        value="3"
    )

with col4:
    st.metric(
        label="Inference",
        value="Live"
    )

st.divider()

# =====================================================
# MAIN WORKSPACE
# =====================================================

left_col, right_col = st.columns(
    [1.5, 1],
    gap="large"
)

with left_col:

    st.header("💬 Analyse a Message")

    st.write(
        "Enter a message below to receive a classification "
        "and probability estimates."
    )

    message = st.text_area(
        "Message",
        placeholder=(
            "Type or paste your message here..."
        ),
        height=200
    )

    analyse = st.button(
        "🔍 Analyse Message",
        type="primary",
        use_container_width=True
    )

    if analyse:

        if not message.strip():

            st.warning(
                "Please enter a message before analysing."
            )

        else:

            with st.spinner(
                "Analysing message..."
            ):

                prediction, probabilities = (
                    moderate_message(message)
                )

            st.session_state.history.insert(
                0,
                {
                    "Time": datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                    "Message": message[:80],
                    "Prediction": prediction.upper()
                }
            )

            st.session_state.history = (
                st.session_state.history[:10]
            )

            st.divider()

            st.subheader("Moderation Result")

            if prediction == "clean":

                st.success(
                    "CLEAN — The model classified this "
                    "message as clean."
                )

            elif prediction == "spam":

                st.warning(
                    "SPAM — The model classified this "
                    "message as spam."
                )

            elif prediction == "abusive":

                st.error(
                    "ABUSIVE — The model classified this "
                    "message as abusive."
                )

            st.metric(
                label="Predicted Category",
                value=prediction.upper()
            )

            st.subheader("📊 Confidence Scores")

            probability_data = pd.DataFrame(
                {
                    "Category": [
                        label.upper()
                        for label in probabilities.keys()
                    ],
                    "Confidence (%)": [
                        round(value * 100, 2)
                        for value in probabilities.values()
                    ]
                }
            )

            st.dataframe(
                probability_data,
                hide_index=True,
                use_container_width=True
            )

            st.subheader("📈 Probability Distribution")

            chart_data = probability_data.set_index(
                "Category"
            )

            st.bar_chart(
                chart_data,
                y="Confidence (%)"
            )

            with st.expander(
                "📝 View Processing Details"
            ):

                st.write("Original Message:")

                st.write(message)

                st.write("Processed Message:")

                st.code(
                    clean_text(message)
                )

                st.info(
                    "Model probabilities are estimates and "
                    "do not guarantee correct classification."
                )

with right_col:

    st.header("⚙️ System Overview")

    st.write(
        "Aegis AI uses a four-stage machine learning "
        "pipeline to analyse text."
    )

    with st.container(border=True):

        st.subheader("01 · Text Preprocessing")

        st.write(
            "Converts text to lowercase, removes punctuation, "
            "and normalises whitespace."
        )

    with st.container(border=True):

        st.subheader("02 · TF-IDF Vectorisation")

        st.write(
            "Converts text into numerical features using "
            "the trained TF-IDF vectorizer."
        )

    with st.container(border=True):

        st.subheader("03 · Classification")

        st.write(
            "Logistic Regression predicts the message category "
            "and estimates class probabilities."
        )

    with st.container(border=True):

        st.subheader("04 · Moderation Output")

        st.write(
            "Displays the predicted category and probability "
            "distribution for the input message."
        )

    st.info(
        "The system supports clean, spam, and abusive "
        "classification."
    )

st.divider()

# =====================================================
# EXAMPLE MESSAGES
# =====================================================

st.header("🧪 Example Messages")

st.write(
    "Copy one of the examples below into the message box."
)

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:

    st.subheader("🟢 Normal")

    st.write(
        "Hello, how are you today?"
    )

with example_col2:

    st.subheader("🟠 Promotional")

    st.write(
        "WIN MONEY NOW!!!"
    )

with example_col3:

    st.subheader("🔴 Abusive")

    st.write(
        "You are stupid and useless."
    )

st.divider()

# =====================================================
# SESSION HISTORY
# =====================================================

st.header("📋 Recent Analysis History")

st.caption(
    "History is retained only during the current session."
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True
    )

else:

    st.info(
        "No messages have been analysed in this session."
    )

st.divider()

# =====================================================
# DEVELOPER SECTION
# =====================================================

st.header("👨‍💻 About the Deployer")

st.subheader("Divya Mohan")

st.write(
    "Undergraduate at IIT Patna, pursuing a BS in "
    "Computer Science and Data Analytics."
)

st.write(
    "Aspiring Machine Learning Engineer and Data Scientist "
    "interested in Natural Language Processing, Artificial "
    "Intelligence, and real-world machine learning applications."
)

st.write(
    "This project demonstrates data preparation, text "
    "preprocessing, TF-IDF vectorisation, model training, "
    "evaluation, and Streamlit deployment."
)

st.subheader("🤝 Connect With Me")

st.write(
    "Let's connect and collaborate on Machine Learning, "
    "Data Science, Artificial Intelligence, and NLP projects."
)

contact_col1, contact_col2, contact_col3 = st.columns(3)

with contact_col1:

    st.link_button(
        "🔗 LinkedIn",
        "https://linkedin.com/in/divya-mohan-a42b12389",
        use_container_width=True
    )

with contact_col2:

    st.link_button(
        "💻 GitHub",
        "https://github.com/divyamohan6388-droid",
        use_container_width=True
    )

with contact_col3:

    st.link_button(
        "✉️ Email Me",
        "mailto:divyamohan6388@gmail.com",
        use_container_width=True
    )

st.divider()

# =====================================================
# FOOTER
# =====================================================

st.caption(
    "Aegis AI | Built with Python, NLP, Scikit-learn, "
    "and Streamlit | © 2026 Divya Mohan"
)