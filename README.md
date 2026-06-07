# Twitter Sentiment Analysis Using Machine Learning

A web-based sentiment analysis application built with Python and Streamlit that classifies tweets as **Positive** or **Negative** using machine learning models trained on the Sentiment140 dataset.

## Features

* Sentiment analysis of tweets in real time
* Text preprocessing and cleaning
* Stopword removal and lemmatization
* Negation handling (e.g., "not good" → "not_good")
* TF-IDF feature extraction using character n-grams
* Multiple machine learning models:

  * Logistic Regression
  * Support Vector Machine (SVM)
* Performance evaluation:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Confusion Matrix visualization
* Interactive Streamlit user interface

## Dataset

This project uses the **Sentiment140** dataset containing 1.6 million labeled tweets.

Sentiment labels:

* `0` → Negative
* `4` → Positive

### Dataset Setup

Download the Sentiment140 dataset and place the file:

```text
training.1600000.processed.noemoticon.csv
```

in the project's root directory.

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn

## Project Structure

```text
Sentiment_Analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── training.1600000.processed.noemoticon.csv
└── screenshots/
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/saleh-khattak/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### Create a Virtual Environment

```bash
python -m venv myenv
```

### Activate the Environment

Windows:

```bash
myenv\Scripts\activate
```

Linux/macOS:

```bash
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## How It Works

1. Load the Sentiment140 dataset
2. Clean and preprocess tweets
3. Convert text into TF-IDF features
4. Train the selected machine learning model
5. Evaluate model performance
6. Predict sentiment for user-entered tweets

## Example

Input:

```text
I absolutely love this project!
```

Output:

```text
Positive 😊
Confidence Score: 0.95
```

## Future Improvements

* Word Cloud visualization
* Deep Learning models (LSTM, GRU)
* BERT-based sentiment analysis
* Model persistence using Joblib
* Streamlit Cloud deployment
* Multi-class sentiment classification

## Author

Muhammad Saleh

GitHub: https://github.com/saleh-khattak

LinkedIn: https://www.linkedin.com/in/muhammad-saleh-a842b434a
