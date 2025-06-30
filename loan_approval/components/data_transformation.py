from dataclasses import dataclass
import os
import sys

from loan_approval.logger import logging
from loan_approval.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder , LabelEncoder
import numpy as np
import pandas as pd
from loan_approval.utils.main_utils import transform_target , save_object


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_object_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                'Loan_Amount_Term', 'Credit_History']
            label_encode_cols = [
                'Gender', 'Married', 'Self_Employed'
            ]

            label_encode_categories = [
                            ['Male', 'Female'],   
                            ['No', 'Yes'],        
                            ['No', 'Yes']         
                        ]


            oridianl_cols = [ 'Education', 'Property_Area','Dependents']
            ordinal_categories = [
                ['Not Graduate','Graduate'],
                ['Rural','Semiurban','Urban'],
                ['0', '1', '2', '3+'] 
            ]

            logging.info("Preprocessing Initiated.")
            preprocessor = ColumnTransformer([
                ('normalization' ,StandardScaler() , numerical_cols ),
                ('label_encoding', OrdinalEncoder(categories=label_encode_categories, dtype=np.int64), label_encode_cols),
                ('ordinal_encoding' , OrdinalEncoder(categories=ordinal_categories  , dtype=np.int64) ,oridianl_cols )
            ])

            return preprocessor
            
        except Exception as e:
            print(CustomException(e , sys))
    
    def initiate_data_transformation(self , train_pth , test_pth):
        try:
            train_df = pd.read_csv(train_pth)
            test_df = pd.read_csv(test_pth)

            logging.info("train and test data loaded...")
            
            preprocessing_obj = self.get_data_transformation()

            X_train = train_df.drop(columns=["Loan_ID" , "Loan_Status"])
            y_train = train_df["Loan_Status"]

            X_test = test_df.drop(columns=["Loan_ID" , "Loan_Status"])
            y_test = test_df["Loan_Status"]

            X_train_arr = preprocessing_obj.fit_transform(X_train)    # Here preprocessor object is geetting trained and fitted.
            X_test_arr = preprocessing_obj.transform(X_test)          # Here we only tranform on the learned preprocessor

            y_train_arr = y_train.map({'Y':1 , 'N':0})
            y_test_arr = y_test.map({'Y':1 , 'N':0})

            logging.info("Applying preproccessing object on training and test data.")

            train_arr = np.c_[X_train_arr ,y_train_arr ]
            test_arr = np.c_[X_test_arr ,y_test_arr ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing object picke file saved")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            print(CustomException(e , sys))


