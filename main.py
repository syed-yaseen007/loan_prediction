import pandas as pd
import numpy as np
# from js import document
import pickle
import warnings

# Disable warnings in the browser
warnings.filterwarnings("ignore")

# Load model
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

def get_predictions():
    data = {
        "ApplicantIncome": input("Applicant Income: "),
        "CoapplicantIncome": input("Coapplicant Income: "),
        "Credit_History": input("Credit History (c_yes/c_no): "),
        "Dependents": input("Dependents (0/1/2/3+): "),
        "Education": input("Education (Graduate/Not Graduate): "),
        "Gender": input("Gender (male/female): "),
        "LoanAmount": input("Loan Amount: "),
        "Loan_Amount_Term": input("Loan Amount Term: "),
        "Married": input("Married (yes/no): "),
        "Property_Area": input("Property Area (Rural/Semiurban/Urban): "),
        "Self_Employed": input("Self Employed (s_yes/s_no): ")
    }

    # Feature engineering
    Gender = 1 if data["Gender"] == "male" else 0
    Married = 1 if data["Married"] == "yes" else 0
    
    if data["Dependents"] == "0":
        Dependents = 0
    elif data["Dependents"] == "1":
        Dependents = 1
    elif data["Dependents"] == "2":
        Dependents = 2
    else:
        Dependents = 3
    
    Education = 0 if data["Education"] == "Graduate" else 1
    Self_Employed = 1 if data["Self_Employed"] == "s_yes" else 0
    LoanAmount = np.log(int(data["LoanAmount"]))
    Loan_Amount_Term = np.log(int(data["Loan_Amount_Term"]))
    Credit_History = 1 if data["Credit_History"] == "c_yes" else 0
    
    if data["Property_Area"] == "Rural":
        Property_Area = 0
    elif data["Property_Area"] == "Semiurban":
        Property_Area = 1
    else:
        Property_Area = 2
    
    TotalIncome = np.log(int(data["ApplicantIncome"]) + int(data["CoapplicantIncome"]))

    # Predict
    predictionData = [Gender, Married, Dependents, Education, Self_Employed,
                      LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, TotalIncome]
    result = loaded_model.predict([predictionData])

    if result[0] == 1:
        result = "will"
    else:
        result = "will not"

    # Show result in the UI
    print(f"The applicant {result} get the loan.")

    return result
