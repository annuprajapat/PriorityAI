import streamlit as st
import pandas as pd
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="PriorityAI", page_icon="🧾", layout="centered")


data = pd.read_csv("complaints.csv")

X = data["text"]
y_priority = data["priority"]
y_category = data["category"]


# VECTORIZATION

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)


# TRAIN MODELS

model_priority = LogisticRegression()
model_priority.fit(X_vector, y_priority)

model_category = LogisticRegression()
model_category.fit(X_vector, y_category)


# UI DESIGN


st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧾 PriorityAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Intelligent Complaint Management System</h4>", unsafe_allow_html=True)

st.markdown("---")

# Input Section
st.subheader(" Enter Complaint")

user_input = st.text_area("Describe your issue:", height=120)


# BUTTON

if st.button("🔍 Analyze Complaint"):
    if user_input.strip() != "":

        input_vector = vectorizer.transform([user_input])

        # Predictions
        pred_priority = model_priority.predict(input_vector)[0]
        pred_category = model_category.predict(input_vector)[0]

        # Confidence
        prob = model_priority.predict_proba(input_vector)
        confidence = max(prob[0])

        st.markdown("---")
        st.subheader("📊 Analysis Result")

        # Stylish Output
        col1, col2 = st.columns(2)

        col1.metric("Priority", pred_priority)
        col2.metric("Category", pred_category)

        st.progress(int(confidence * 100))
        st.write(f"Confidence Score: {round(confidence*100,2)}%")

        # Explanation
        if pred_priority == "High":
            st.error("⚠️ Immediate attention required!")
        elif pred_priority == "Medium":
            st.warning("🟡 Needs attention soon.")
        else:
            st.success("🟢 Low priority issue.")

        
        # DOWNLOAD REPORT
        
        report = f"""
Complaint Report
-------------------------
Complaint: {user_input}

Predicted Priority: {pred_priority}
Category: {pred_category}
Confidence: {round(confidence*100,2)}%

Generated on: {datetime.datetime.now()}
"""

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="complaint_report.txt",
            mime="text/plain"
        )

    else:
        st.warning("⚠️ Please enter a complaint first!")


# FOOTER

st.markdown("---")
st.markdown(
    "<center><small>Built using Machine Learning & NLP (TF-IDF + Logistic Regression)</small></center>",
    unsafe_allow_html=True
)