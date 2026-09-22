
# 🛡️ Aegis AI — Intelligent Content Moderation & Spam Detection

<p align="center">
  <b>Real-Time Machine Learning System for Content Moderation and Spam Detection</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Project Status">
</p>

---

## 📌 Overview

**Aegis AI** is a machine learning-powered content moderation and spam detection application designed to classify user-generated text into three categories:

- 🟢 **Clean** — Non-harmful and acceptable content
- 🔴 **Abusive** — Potentially offensive or abusive content
- 🟠 **Spam** — Promotional, misleading, or unwanted messages

The application combines natural language processing (NLP), TF-IDF feature extraction, and Logistic Regression to analyse text and provide classification results through an interactive Streamlit interface.

This project explores how machine learning can support **Trust & Safety systems, online communities, and automated content moderation workflows**.

---

## 🚀 Live Demo

🔗 **Streamlit App:** [Add Your Deployed App URL Here]

> The live demo link will be added after deployment.

---

## 🎯 Project Objectives

- Build a text classification system for content moderation.
- Detect spam and potentially abusive messages.
- Apply natural language processing techniques to text data.
- Train and evaluate a multiclass machine learning model.
- Display prediction results and confidence scores.
- Develop an interactive web interface using Streamlit.
- Explore the role of ML in Trust & Safety applications.

---

## 🧠 Machine Learning Pipeline

```text
Raw Text Input
      │
      ▼
Text Preprocessing
      │
      ├── Lowercasing
      ├── Punctuation Removal
      └── Whitespace Normalisation
      │
      ▼
TF-IDF Feature Extraction
      │
      ▼
Logistic Regression Model
      │
      ▼
Text Classification
      │
      ├── Clean
      ├── Abusive
      └── Spam
      │
      ▼
Prediction & Confidence Scores
      │
      ▼
Streamlit Web Application
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Multiclass text classification |
| Joblib | Model and vectorizer serialisation |
| Streamlit | Interactive web application |
| Matplotlib | Prediction probability visualisation |

---

## 📂 Dataset

The project combines text data from two sources:

### 1. SMS Spam Dataset

Used to identify spam and non-spam messages.

### 2. Toxic Comment Dataset

Used to identify potentially abusive or toxic comments.

The datasets were cleaned and transformed into a unified format containing:

| Column | Description |
|---|---|
| `text` | Original user-generated text |
| `label` | Classification category |
| `clean_text` | Preprocessed text |

### Dataset Processing

- Combined relevant datasets.
- Standardised label names.
- Removed unnecessary columns.
- Applied text preprocessing.
- Created a unified classification dataset.
- Split the data into training and testing sets.

> Dataset licensing and usage terms should be reviewed before redistribution.

---

## ⚙️ Model Development

### Text Preprocessing

The preprocessing pipeline includes:

1. Converting text to lowercase.
2. Removing punctuation.
3. Normalising whitespace.
4. Preparing text for feature extraction.

### Feature Engineering

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical features.

Configuration:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

The model uses both unigrams and bigrams to capture individual words and short word combinations.

### Classification Model

The primary model is Logistic Regression with balanced class weights:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)
```

Balanced class weights help the model account for differences in class distribution during training.

---

## 📊 Model Performance

The model was evaluated using a held-out test dataset.

| Metric | Result |
|---|---|
| Training Samples | 131,790 |
| Testing Samples | 32,948 |
| TF-IDF Features | 10,000 |
| Initial Logistic Regression Accuracy | Approximately 92.18% |
| Macro F1-Score | Approximately 0.7781 |
| Macro Recall | Approximately 0.8953 |

> Performance depends on the dataset, preprocessing pipeline, train-test split, and model configuration. These metrics are experimental results from the current implementation and should not be treated as production safety guarantees.

### Important Evaluation Considerations

For a real-world moderation system, accuracy alone is insufficient. Additional evaluation should include:

- Precision and recall for each class.
- Confusion matrix analysis.
- False positive rate.
- False negative rate.
- Performance on unseen and evolving content.
- Human review of uncertain predictions.

---

## 💻 Application Features

### 🔍 Real-Time Text Classification

Analyse a message and receive a predicted moderation category.

### 📈 Confidence Analysis

View the model's estimated class probabilities.

### 🧾 Prediction History

Review previously analysed messages during the application session.

### 📊 Probability Visualisation

Compare the model's predicted probabilities across the available categories.

### 🖥️ Interactive Dashboard

Use the Streamlit interface to test messages without writing additional code.

---

## 🗂️ Project Structure

```text
Aegis-AI/
│
├── data/
│   ├── tfidf_vectorizer.pkl
│   └── logistic_regression_model.pkl
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> Update the notebook names and folder structure to match the actual files in your repository.

---

## ▶️ Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
scikit-learn
pandas
numpy
scipy
joblib
matplotlib
```

Use the dependency versions compatible with the environment in which your models were trained.

---

## 🔮 Future Improvements

- [ ] Add a confidence threshold for uncertain predictions.
- [ ] Introduce a human-in-the-loop moderation workflow.
- [ ] Improve performance on borderline and ambiguous messages.
- [ ] Experiment with advanced NLP models such as transformer-based architectures.
- [ ] Add multilingual content moderation.
- [ ] Implement model monitoring and drift detection.
- [ ] Add explainability techniques for model predictions.
- [ ] Create a feedback mechanism for reviewed classifications.
- [ ] Evaluate the model on a separate real-world test dataset.
- [ ] Add automated testing and CI/CD.
- [ ] Improve protection against adversarial and obfuscated text.

---

## ⚠️ Limitations

- Model predictions are not guaranteed to be correct.
- Confidence scores represent model estimates and are not human-verified certainty.
- The model may produce false positives or false negatives.
- Training data may not represent every language, community, or communication style.
- The current system should not be used as the sole decision-maker for high-impact moderation actions.
- Production deployment would require additional privacy, security, fairness, and safety evaluation.

---

## 🔐 Responsible AI Considerations

Content moderation systems can affect how people communicate online. A responsible production system should consider:

- User privacy and data protection.
- Bias across different groups and language varieties.
- Transparent moderation policies.
- Human review for uncertain or high-impact decisions.
- Regular evaluation using representative data.
- Secure handling of user-generated content.

Aegis AI is an educational and portfolio project exploring machine learning-based text classification.

---

## 👨‍💻 Developer

### Divya

**Undergraduate at IIT Patna | Aspiring Data Scientist & ML Engineer**

Interested in:

- Machine Learning
- Data Science
- Natural Language Processing
- Statistical Modelling
- Trust & Safety Systems
- Artificial Intelligence

<p align="left">
  <a href="https://linkedin.com/in/divya-mohan-a42b12389">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="https://github.com/divyamohan6388-droid">
    <img src="https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github" alt="GitHub">
  </a>
</p>

📧 Email: divyamohan6388@gmail.com

---

## ⭐ Acknowledgements

This project was developed as part of my practical learning journey in:

- Machine Learning
- Natural Language Processing
- Data Analytics
- Model Evaluation
- Streamlit Application Development

---

<p align="center">
  Built with Python, Machine Learning, and curiosity.
</p>