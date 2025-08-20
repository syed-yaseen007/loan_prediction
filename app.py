import streamlit as st
import numpy as np
import pickle
import warnings
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Disable warnings
warnings.filterwarnings("ignore")

# Load trained model
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Loan Prediction App", page_icon="🏦", layout="centered")

# Title
st.title("🏦 Loan Prediction App")
st.markdown("### Predict your loan eligibility in seconds")

with st.expander("ℹ️ About this App"):
    st.write("""
    This app uses a Machine Learning model trained on the 
    [Kaggle Loan Prediction Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset).
    
    ⚠️ Disclaimer: The prediction is **not 100% accurate** and should only be used for educational purposes.
    """)

st.divider()

# ---------------------------
# Input Form (with defaults)
# ---------------------------
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"], index=0)
    married = st.radio("Marital Status", ["Yes", "No"], index=0)
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
    education = st.radio("Education", ["Graduate", "Not Graduate"], index=0)
    self_employed = st.radio("Self Employed", ["No", "Yes"], index=0)

with col2:
    applicant_income_lpa = st.number_input("Applicant Income (in LPA)", min_value=0.0, step=0.5, value=6.0)
    coapplicant_income_lpa = st.number_input("Coapplicant Income (in LPA)", min_value=0.0, step=0.5, value=1.5)
    loan_amount_lakh = st.number_input("Loan Amount (in Lakhs)", min_value=0.0, step=0.5, value=5.0)
    loan_term_years = st.number_input("Loan Term (in Years)", min_value=1, max_value=30, step=1, value=20)
    credit_history = st.radio("Previous Loans Cleared?", ["Yes", "No"], index=0)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)

st.divider()

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔮 Predict Loan Approval", use_container_width=True):

    # ✅ Convert values to dataset scale
    applicant_income = int((applicant_income_lpa * 100000) / 12)   # LPA → monthly
    coapplicant_income = int((coapplicant_income_lpa * 100000) / 12)
    loan_amount = int(loan_amount_lakh * 100)   # Lakh → thousands
    loan_amount_term = int(loan_term_years * 12)  # years → months

    # Encode categorical features
    Gender = 1 if gender == "Male" else 0
    Married = 1 if married == "Yes" else 0
    
    if dependents == "0":
        Dependents = 0
    elif dependents == "1":
        Dependents = 1
    elif dependents == "2":
        Dependents = 2
    else:
        Dependents = 3
    
    Education = 0 if education == "Graduate" else 1
    Self_Employed = 1 if self_employed == "Yes" else 0
    LoanAmount = np.log(max(loan_amount, 1))
    Loan_Amount_Term = np.log(max(loan_amount_term, 1))
    Credit_History = 1 if credit_history == "Yes" else 0
    
    if property_area == "Rural":
        Property_Area = 0
    elif property_area == "Semiurban":
        Property_Area = 1
    else:
        Property_Area = 2
    
    TotalIncome = np.log(max(applicant_income + coapplicant_income, 1))

    # Prepare final data
    predictionData = [
        Gender, Married, Dependents, Education, Self_Employed,
        LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, TotalIncome
    ]

    # Model prediction
    result = loaded_model.predict([predictionData])
    probability = loaded_model.predict_proba([predictionData])[0][1]  # probability of approval

    # ---------------------------
    # Output
    # ---------------------------
    st.subheader("📊 Prediction Result")

    if result[0] == 1:
        verdict = "✅ Congratulations! The model predicts that you WILL get the loan."
        st.success(verdict)
        st.info(f"Model confidence: **{probability*100:.2f}%**")
    else:
        verdict = "❌ Sorry! The model predicts that you WILL NOT get the loan."
        st.error(verdict)
        st.info(f"Model confidence: **{(1-probability)*100:.2f}%**")

    # ---------------------------
    # Generate PDF Report
    # ---------------------------
    def generate_pdf():
        file_path = "loan_report.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("🏦 Loan Prediction Report", styles['Title']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Applicant Details:</b>", styles['Heading2']))
        elements.append(Paragraph(f"Gender: {gender}", styles['Normal']))
        elements.append(Paragraph(f"Marital Status: {married}", styles['Normal']))
        elements.append(Paragraph(f"Dependents: {dependents}", styles['Normal']))
        elements.append(Paragraph(f"Education: {education}", styles['Normal']))
        elements.append(Paragraph(f"Self Employed: {self_employed}", styles['Normal']))
        elements.append(Paragraph(f"Applicant Income: {applicant_income_lpa} LPA", styles['Normal']))
        elements.append(Paragraph(f"Coapplicant Income: {coapplicant_income_lpa} LPA", styles['Normal']))
        elements.append(Paragraph(f"Loan Amount: {loan_amount_lakh} Lakhs", styles['Normal']))
        elements.append(Paragraph(f"Loan Term: {loan_term_years} Years", styles['Normal']))
        elements.append(Paragraph(f"Credit History: {credit_history}", styles['Normal']))
        elements.append(Paragraph(f"Property Area: {property_area}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Prediction Result:</b>", styles['Heading2']))
        elements.append(Paragraph(verdict, styles['Normal']))
        elements.append(Paragraph(f"Confidence Score: {probability*100:.2f}%", styles['Normal']))

        doc.build(elements)
        return file_path

    pdf_file = generate_pdf()
    with open(pdf_file, "rb") as f:
        st.download_button("📥 Download PDF Report", f, file_name="Loan_Prediction_Report.pdf", mime="application/pdf")


