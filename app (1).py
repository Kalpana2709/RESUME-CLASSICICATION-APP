import streamlit as st
import pickle
import numpy as np
from docx import Document
import fitz  # PyMuPDF
import os

# Title
st.title("🧠 AI Resume Classifier App")
st.write("Upload a resume (DOCX or PDF) and get an instant prediction.")

# Load model and vectorizer
with open("resume_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Functions to read files
def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Upload file
uploaded_file = st.file_uploader("Upload Resume (.docx or .pdf)", type=["docx", "pdf"])

if uploaded_file:
    # Extract text
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension == ".docx":
        resume_text = read_docx(uploaded_file)
    elif file_extension == ".pdf":
        resume_text = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a DOCX or PDF.")
        resume_text = None

    # Predict
    if resume_text:
        # Vectorize
        input_vector = tfidf_vectorizer.transform([resume_text])
        prediction = model.predict(input_vector)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Output
        st.subheader("✅ Predicted Category:")
        st.success(predicted_label)

