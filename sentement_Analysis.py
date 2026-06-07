import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# NLTK Downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Streamlit Config

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    layout="wide"
)

st.title("Twitter Sentiment Analysis Using Machine Learning")
st.write("**Dataset:** Sentiment140 (1.6M Tweets)")
st.markdown("---")


# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv(
        "training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        header=None
    )
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['sentiment', 'text']]
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
    return df.sample(250000, random_state=42)

df = load_data()


# Data Distribution

st.subheader("📊 Data Distribution")

fig, ax = plt.subplots(figsize=(4,2))
sns.countplot(x=df['sentiment'], ax=ax)
ax.set_xticklabels(['Negative', 'Positive'])
plt.tight_layout()
st.pyplot(fig, use_container_width=False)


# Text Preprocessing

st.subheader("🧹 Text Preprocessing")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)

    # Handle negation
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)

    text = re.sub(r"[^a-z_\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

with st.spinner("Cleaning tweets..."):
    df['clean_text'] = df['text'].apply(clean_text)

st.success("Text preprocessing completed!")

st.write("### Sample Cleaned Tweets")
st.dataframe(df[['text', 'clean_text']].head())


# Feature Extraction

st.subheader("🔢 Feature Extraction (TF-IDF)")

tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    min_df=5
)


X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model Selection

st.subheader("🤖 Model Training")

model_choice = st.selectbox(
    "Choose Machine Learning Model",
    ["Logistic Regression", "Support Vector Machine (SVM)"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression(
        C=3.0,
        solver="liblinear",
        max_iter=1000
    )
else:
    model = LinearSVC(C=2.0)

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

st.success("Model trained successfully!")


# Evaluation

st.subheader("📈 Model Evaluation")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")


# Confusion Matrix

st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots(figsize=(4,2.2))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
plt.tight_layout()
st.pyplot(fig2, use_container_width=False)


# User Tweet Prediction

st.subheader("✍️ Predict Sentiment of Your Tweet")

user_input = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        if model_choice == "Logistic Regression":
            confidence = np.max(model.predict_proba(vector))
        else:
            confidence = abs(model.decision_function(vector)[0])
            confidence = confidence / (confidence + 1)

        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence Score:** {confidence:.2f}")
