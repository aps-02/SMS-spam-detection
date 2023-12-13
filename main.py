from flask import Flask, render_template, request
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import nltk

app = Flask(__name__)


nltk.download("punkt")
nltk.download("stopwords")

data = pd.read_csv(
    r"C:/Users/DELL/Desktop/SMS spam detection/dataset/spam.csv", encoding="ISO-8859-1"
).sample(frac=0.03, random_state=1)


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [
        stemmer.stem(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(words)


data["clean_text"] = data["v2"].apply(preprocess_text)
data["label"] = data["v1"]


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data["clean_text"])
y = data["label"]
oversample = RandomOverSampler(sampling_strategy="minority")
X_over, y_over = oversample.fit_resample(X, y)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_over, y_over)


def is_spam(message):
    if not message.strip():
        return "Please provide a message"
    processed = preprocess_text(message)
    transformed = tfidf.transform([processed])
    prediction = rf_model.predict(transformed)
    return "spam" if prediction[0] == "spam" else "not spam"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    message = request.form.get("message")
    if not message:
        return render_template("result.html", result="No message provided")
    result = is_spam(message)
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
