
import streamlit as st
import pickle
import docx2txt
import fitz  # PyMuPDF
import os

# Load models
with open("resume_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Function to extract text from uploaded file
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        doc = fitz.open(stream=file.read(), filetype='pdf')
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif ext == 'docx':
        with open("temp.docx", "wb") as f:
            f.write(file.read())
        return docx2txt.process("temp.docx")
    else:
        return ""

# Streamlit UI
st.set_page_config(page_title="Resume Classifier", page_icon="🧠")
st.title("🧠 AI Resume Classifier")
st.markdown("Upload a resume to predict the candidate category")

uploaded_file = st.file_uploader("Choose a resume file (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)
    
    if resume_text.strip() == "":
        st.error("❌ Failed to extract text from the resume.")
    else:
        st.success("✅ Text extracted successfully!")
        st.subheader("📄 Extracted Text Preview:")
        st.write(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

        # Preprocess & predict
        features = vectorizer.transform([resume_text])
        pred = model.predict(features)
        label = label_encoder.inverse_transform(pred)

        st.subheader("🎯 Predicted Category:")
        st.success(label[0])
