import sys

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.sklearn
from loan_approval.exception import CustomException
from loan_approval.logger import logging
from sklearn.metrics import accuracy_score , precision_score , f1_score , recall_score
import os
from urllib.parse import urlparse
from loan_approval.utils.main_utils import load_object
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
load_dotenv()

class ModelEvaluation:
    def __init__(self):
        logging.info("model evaluation started")
    
    def get_eval_metrics(self  , y_true , y_pred):
        accuracy = accuracy_score(y_true=y_true , y_pred=y_pred)
        precision = precision_score(y_true=y_true , y_pred=y_pred)
        f1 = f1_score(y_true=y_true , y_pred=y_pred)
        recall = recall_score(y_true=y_true , y_pred=y_pred)

        return (accuracy , precision , f1 , recall)
    
    def initiate_model_evaluation(self , train_arr , test_arr ):
        try:
            X_test , y_test , X_train , y_train = (
                test_arr[: , :-1],
                test_arr[: , -1],

                train_arr[: , :-1],
                train_arr[: , -1]

            )

            model_path = os.path.join("artifacts" , "model.pkl")
            model = load_object(model_path)

            preprocessor_pth = os.path.join("artifacts" , "preprocessor.pkl")
            preprocessor = load_object(preprocessor_pth)

            logging.info("Model has been loaded")
            mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment("Loan Approval Model Training")
            url_type = urlparse(mlflow.get_tracking_uri()).scheme
            print(url_type)

            input_example = X_train[:5]
            signature = infer_signature(X_test , model.predict(X_test))

            with mlflow.start_run():
                prediction = model.predict(X_test)
                accuracy , precision , f1 , recall = self.get_eval_metrics(y_true=y_test , y_pred=prediction) 

                mlflow.log_metric(key="accuracy" , value=accuracy)
                mlflow.log_metric(key="precision" , value=precision)
                mlflow.log_metric(key="f1_score" , value=f1)
                mlflow.log_metric(key="recall" , value=recall)

                # Saving model in mlflow
                pipe = Pipeline([
                    ('preprocessor' , preprocessor),
                    ('model' , model)
                ])

                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name="best_model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name="Loan Approval Modal",


                )


        except Exception as e:
            print(CustomException(e , sys))

