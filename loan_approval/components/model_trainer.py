from dataclasses import dataclass
import os
import sys
from imblearn.pipeline import Pipeline as Imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from loan_approval.exception import CustomException
from loan_approval.utils.main_utils import evaluate_model, save_object
from loan_approval.logger import logging



@dataclass
class ModelTrainerConfig:
        model_object_path:str = os.path.join('artifacts' , 'model.pkl')
        model_params:str = os.path.join('artifacts' , 'params.json')

class ModelTrainer:
        def __init__(self):
                self.model_config = ModelTrainerConfig()
        
        def initiate_model_training(self , train_arr , test_arr):
            try:
                X_train , X_test , y_train , y_test = (
                      train_arr[: , :-1],
                      test_arr[: , :-1],
                      train_arr[: , -1],
                      test_arr[: , -1]
                )
            
                models = {
                    "RandomForest": RandomForestClassifier(
                          max_depth=10, 
                          max_features='sqrt',
                          class_weight='balanced_subsample',
                          min_samples_leaf=4,
                          min_samples_split=13,
                          n_estimators=410
                    ),
                    "LogisticRegression":LogisticRegression(),
                    "NaiveBayes":GaussianNB(),
                }
                model_report = evaluate_model(
                      X_train=X_train,
                      X_test=X_test,
                      y_train=y_train,
                      y_test=y_test,
                      models=models
                )
                
                print("="*40)
                print(model_report)
                best_model_name = model_report['Model'][0]
                best_model_score = model_report['F1 Score'][0]
                logging.info(f"{best_model_name} performed better than others")

                best_model = models[best_model_name]

                print(f'Best Model Found , Model Name : {best_model_name} , F1 Score : {best_model_score}')

                save_object(
                      file_path=self.model_config.model_object_path,
                      obj = best_model
                )



            except Exception as e:
                  print(CustomException(e , sys))
        
