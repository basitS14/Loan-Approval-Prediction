from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from datetime import datetime, timedelta

from loan_approval.components.data_ingestion import DataIngestion
from loan_approval.components.data_transformation import DataTransformation
from loan_approval.components.model_trainer import ModelTrainer
from loan_approval.components.model_evaluation import ModelEvaluation

default_args = {
    'owner': 'basit',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'loan_approval_training_pipeline',
    default_args=default_args,
    description='Loan approval ML pipeline using Airflow',
    schedule_interval=None,  # can set to '@daily' etc.
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['loan', 'ml', 'training'],
) as dag:

    def run_data_ingestion(**kwargs):
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        kwargs['ti'].xcom_push(key='train_path', value=train_path)
        kwargs['ti'].xcom_push(key='test_path', value=test_path)

    def run_data_transformation(**kwargs):
        ti = kwargs['ti']
        train_path = ti.xcom_pull(key='train_path')
        test_path = ti.xcom_pull(key='test_path')
        obj = DataTransformation()
        train_arr, test_arr = obj.initiate_data_transformation(train_path, test_path)
        ti.xcom_push(key='train_arr', value=train_arr)
        ti.xcom_push(key='test_arr', value=test_arr)

    def run_model_training(**kwargs):
        ti = kwargs['ti']
        train_arr = ti.xcom_pull(key='train_arr')
        test_arr = ti.xcom_pull(key='test_arr')
        obj = ModelTrainer()
        obj.initiate_model_training(train_arr, test_arr)

    def run_model_evaluation(**kwargs):
        ti = kwargs['ti']
        train_arr = ti.xcom_pull(key='train_arr')
        test_arr = ti.xcom_pull(key='test_arr')
        obj = ModelEvaluation()
        obj.initiate_model_evaluation(train_arr, test_arr)

    t1 = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_data_ingestion,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id='data_transformation',
        python_callable=run_data_transformation,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
        provide_context=True,
    )

    t4 = PythonOperator(
        task_id='model_evaluation',
        python_callable=run_model_evaluation,
        provide_context=True,
    )

    # Task pipeline
    t1 >> t2 >> t3 >> t4
