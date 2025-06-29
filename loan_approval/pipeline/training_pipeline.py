import os

from loan_approval.components.data_ingestion import DataIngestion
from loan_approval.components.data_transformation import DataTransformation
from loan_approval.components.model_evaluation import ModelEvaluation
from loan_approval.components.model_trainer import ModelTrainer

obj = DataIngestion()
data_trandformation_obj = DataTransformation()
model_trainer_obj = ModelTrainer()
model_eval_obj = ModelEvaluation()


train_path  ,test_path = obj.initiate_data_ingestion()
train_arr , test_arr = data_trandformation_obj.initiate_data_transformation(train_path  ,test_path)

model_trainer_obj.initiate_model_training(train_arr , test_arr)
model_eval_obj.initiate_model_evaluation(train_arr , test_arr)
