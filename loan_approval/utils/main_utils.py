from typing import List
from pandas import DataFrame
import sys
from pandas import Series
import numpy as np

import yaml
from loan_approval.exception import CustomException
from loan_approval.logger import logging
import os 
import pickle
from imblearn.over_sampling import SMOTE

import pandas as pd

from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score




def clean_data(df:DataFrame) -> DataFrame:
    try:
        logging.info("Data cleaning initiated")
        
        df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
        df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
        df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
        df['Self_Employed'] = df["Self_Employed"].fillna(df["Dependents"]).mode()[0]
        df['Loan_Amount_Term'] = df["Loan_Amount_Term"].fillna(df['Loan_Amount_Term'].mode()[0])
        df['Credit_History'] = df["Credit_History"].fillna(df['Credit_History'].mode()[0])
        df['LoanAmount'] = df["LoanAmount"].fillna(df['LoanAmount'].median())
        
        logging.info("Data cleaning completed")

        return df

    except Exception as e:
        print(CustomException(e , sys))



def transform_target(target : Series ) -> Series:
   return target.where((target == "Y") , 1 , 0)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys) 

# def save_params(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)

#         os.makedirs(dir_path, exist_ok=True)

#         with open(file_path, "wb") as file_obj:
#             pickle.dump(obj, file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)

def evaluate_model(X_train , X_test , y_train , y_test , models ):
    results = []
    logging.info("Model training started")
    for model_name , model in models.items():

        model.fit(X_train , y_train)

        y_pred = model.predict(X_test)

        # Append metrics to the results list
        results.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    logging.info("Model report generated")

    # Sort by F1 Score or any other metric
    return results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)


def load_params(params_pth:str) -> dict:

    try:
        with open(params_pth, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        print(CustomException(e , sys))