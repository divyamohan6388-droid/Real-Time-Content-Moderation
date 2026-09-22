# Real-Time Content Moderation & Spam Detection

A machine learning-based web application that classifies text into three categories: clean, abusive, and spam. The project combines natural language processing, TF-IDF feature extraction, and Logistic Regression, with an interactive interface built using Streamlit.

The goal of this project is to explore how machine learning can support content moderation and Trust & Safety systems.

## Live Demo

**Try the application:**  
https://divya-aegis-ai.streamlit.app/

**Source Code:**  
https://github.com/divyamohan6388-droid/Real-Time-Content-Moderation

## Project Overview

Online platforms receive large amounts of user-generated content, making manual moderation difficult to scale. This project demonstrates a basic machine learning pipeline for identifying potentially problematic text.

The application classifies input messages into the following categories:

- **Clean:** Non-abusive and non-spam content
- **Abusive:** Potentially offensive or harmful language
- **Spam:** Unwanted, promotional, or suspicious messages

The predictions are displayed through a Streamlit web interface along with estimated class probabilities.

## Features

- Real-time text classification
- Three-class content moderation
- Spam and abusive content detection
- TF-IDF-based text feature extraction
- Logistic Regression classification
- Class probability visualisation
- Interactive Streamlit dashboard
- Session-based prediction history

## Machine Learning Workflow

The project follows these steps:

1. Collect and combine text datasets.
2. Standardise the labels into a common classification format.
3. Clean the text by converting it to lowercase, removing punctuation, and normalising whitespace.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Evaluate the model on a held-out test dataset.
7. Integrate the trained model into a Streamlit application.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Machine learning and evaluation
- **TF-IDF** — Text feature extraction
- **Logistic Regression** — Text classification
- **Joblib** — Saving and loading trained models
- **Streamlit** — Web application development
- **Matplotlib** — Visualisation


## Dataset

This project uses two publicly available datasets for text classification.

### 1. SMS Spam Collection Dataset

Used to identify spam and legitimate messages.

- **Source:** Kaggle
- **Link:** https://www.kaggle.com/uciml/sms-spam-collection-dataset

### 2. Jigsaw Toxic Comment Classification Challenge

Used to identify potentially abusive or toxic comments.

- **Source:** Kaggle
- **Link:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The datasets were cleaned and combined into a unified classification format containing text and labels.

The original datasets are not included in this repository. The trained TF-IDF vectoriser and Logistic Regression model files are stored in the `data` directory.

Dataset licensing and usage conditions should be reviewed before redistributing or using the original datasets.

## Model Configuration

### TF-IDF Vectorisation

The text was converted into numerical features using TF-IDF with the following configuration:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

The model uses unigrams and bigrams to capture individual words and short word combinations.

### Logistic Regression

The primary classification model was configured as follows:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)
```

Balanced class weights were used to account for differences in class distribution during training.

## Model Performance

The current implementation was evaluated on a held-out test dataset.

| Metric | Result |
|---|---:|
| Training samples | 131,790 |
| Testing samples | 32,948 |
| TF-IDF features | 10,000 |
| Logistic Regression accuracy | Approximately 92.18% |
| Macro F1-score | Approximately 0.7781 |
| Macro recall | Approximately 0.8953 |

These results reflect the current experimental implementation. Model performance may change depending on the dataset split, preprocessing, and training configuration.

Accuracy alone is not sufficient for evaluating a production content moderation system. Precision, recall, false positives, false negatives, and performance across different types of content should also be considered.

## Project Structure

```text
Real-Time-Content-Moderation/
│
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
│
└── data/
    ├── tfidf_vectorizer.pkl
    └── logistic_regression_model.pkl
```

The repository does not include the original CSV datasets or NPZ files.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/divyamohan6388-droid/Real-Time-Content-Moderation.git
```

### 2. Move into the project directory

```bash
cd Real-Time-Content-Moderation
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your default browser.

## Limitations

This project is an educational and portfolio implementation of machine learning-based content classification.

Some limitations include:

- Predictions may contain false positives and false negatives.
- The model may not perform consistently on unfamiliar or obfuscated text.
- The training data may not represent every language, community, or communication style.
- Probability scores are model estimates and should not be interpreted as verified certainty.
- The system should not be used as the sole decision-maker for high-impact moderation decisions.
- Further testing is required before considering a production deployment.

## Future Improvements

- Evaluate the model using a separate, unseen dataset.
- Improve performance on ambiguous and borderline messages.
- Add a confidence threshold for uncertain predictions.
- Experiment with transformer-based NLP models.
- Add multilingual content moderation.
- Introduce human review for uncertain predictions.
- Add model monitoring and drift detection.
- Implement model explainability.
- Add automated testing and continuous integration.
- Improve robustness against modified or obfuscated text.

## Responsible AI Considerations

Content moderation models can affect how people communicate online. A production-oriented system should consider:

- Privacy and secure handling of user-generated content
- Bias and fairness across different groups
- Transparent moderation policies
- Human review for uncertain cases
- Regular model evaluation
- Appropriate handling of false positives and false negatives

This project is intended to demonstrate the technical foundations of text classification and its potential application in Trust & Safety workflows.

## Author

**Divya Mohan**

Undergraduate at IIT Patna  
Aspiring Data Scientist and Machine Learning Engineer

Areas of interest:

- Machine Learning
- Data Science
- Natural Language Processing
- Statistics
- Trust & Safety Systems
- Artificial Intelligence

### Connect with me

- **GitHub:** https://github.com/divyamohan6388-droid
- **LinkedIn:** https://linkedin.com/in/divya-mohan-a42b12389

---

If you found this project useful, feel free to explore the repository and share feedback.
