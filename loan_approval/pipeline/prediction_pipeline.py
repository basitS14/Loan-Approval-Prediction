from json import load
import os
import sys

from loan_approval.logger import logging
from loan_approval.exception import CustomException
from loan_approval.utils.main_utils import load_object
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self , features):
       try:
            preprocessor_pth = os.path.join("artifacts" , "preprocessor.pkl")
            model_pth = os.path.join("artifacts" , "model.pkl")

            preprocessor = load_object(preprocessor_pth)
            model = load_object(model_pth)

            transformed_fea = preprocessor.transform(features)
            pred = model.predict(transformed_fea)

            return pred

       except Exception as e:
           CustomException(e , sys)

class CustomData:
    def __init__(self,
                 Gender: str,
                 Married: str,
                 Dependents: str,
                 Education: str,
                 Self_Employed: str,
                 ApplicantIncome: int,
                 CoapplicantIncome: float,
                 LoanAmount: float,
                 Loan_Amount_Term: float,
                 Credit_History: float,
                 Property_Area: str):
        
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender': [self.Gender],
                'Married': [self.Married],
                'Dependents': [self.Dependents],
                'Education': [self.Education],
                'Self_Employed': [self.Self_Employed],
                'ApplicantIncome': [self.ApplicantIncome],
                'CoapplicantIncome': [self.CoapplicantIncome],
                'LoanAmount': [self.LoanAmount],
                'Loan_Amount_Term': [self.Loan_Amount_Term],
                'Credit_History': [self.Credit_History],
                'Property_Area': [self.Property_Area],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.error('Exception occurred in get_data_as_dataframe')
            raise CustomException(e, sys)
        
