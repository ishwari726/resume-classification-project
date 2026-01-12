import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from docx import Document
from PyPDF2 import PdfReader

# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Load trained model & vectorizer
# -----------------------------
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# NLP tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


CATEGORY_NAME_MAP = {
    "sql developer lightning insight": "SQL Developer",
    "react developer resumes": "React Developer",
    "workday resume": "Workday",
    "workday resumes": "Workday",
    "peoplesoft resumes": "Peoplesoft"
}

# -----------------------------
# Text Cleaning (SAME as training)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)      # remove emails
    text = re.sub(r'http\S+', '', text)      # remove URLs
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # remove numbers & special chars
    text = re.sub(r'\s+', ' ', text)

    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)



# -----------------------------
# Read DOCX file
# -----------------------------
def read_docx(file):
    doc = Document(file)
    return " ".join(p.text for p in doc.paragraphs)

# -----------------------------
# Read PDF file 
# -----------------------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Resume Classifier", layout="centered")

st.markdown(
    """
    <marquee style="color:red; font-weight:bold; font-size:16px;">
    ⚠️ This system supports ONLY SQL Developer, React Developer, Workday & Peoplesoft resumes
    </marquee>
    """,
    unsafe_allow_html=True
)

st.title("📄 Resume Classification System")
st.write("Upload one or more resumes (PDF or DOCX) to predict job category, skills, and confidence score.")

uploaded_files = st.file_uploader(
    "Upload Resume(s)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)


if st.button("Predict"):
    if not uploaded_files:
        st.warning("Please upload at least one resume file.")
    else:
        results = []

        for uploaded_file in uploaded_files:

            # Read file
            if uploaded_file.name.endswith(".pdf"):
                resume_text = read_pdf(uploaded_file)
            else:
                resume_text = read_docx(uploaded_file)

            if resume_text.strip() == "":
                continue

            cleaned_text = clean_text(resume_text)
            vectorized_text = vectorizer.transform([cleaned_text])

            raw_prediction = model.predict(vectorized_text)[0]
            raw_prediction = raw_prediction.strip().lower()
            display_prediction = CATEGORY_NAME_MAP.get(raw_prediction, raw_prediction.title())
            probabilities = model.predict_proba(vectorized_text)[0]
            confidence = max(probabilities) * 100

            results.append({
                "File Name": uploaded_file.name,
                "Predicted Category": display_prediction,
                "Confidence (%)": round(confidence, 2),
            })

        # Convert to table
        results_df = pd.DataFrame(results)

        st.subheader("📊 Resume Classification Results")
        st.dataframe(results_df, use_container_width=True)
