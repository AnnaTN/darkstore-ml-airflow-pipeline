
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
import pandas as pd
import numpy as np
import boto3
from catboost import CatBoostRegressor
import logging
import io
import pickle


from preprocessing import preprocess_data



S3_BUCKET = Variable.get("s3_bucket_name", default_var="your-bucket-name")
S3_ACCESS_KEY = Variable.get("s3_access_key", default_var=None)
S3_SECRET_KEY = Variable.get("s3_secret_key", default_var=None)
S3_MODEL_KEY = Variable.get("s3_model_key", default_var="catboost_model.pkl")

POSTGRES_CONN_ID = "postgres_sales_db"

RANDOM_STATE = 42
CAT_FEATURES = ['store', 'dept', 'is_holiday', 'type']

logger = logging.getLogger(__name__)




def load_data_from_postgres(**context):

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    create_inference_data_temp = """
    DROP TABLE IF EXISTS inference_data_temp;

    CREATE TABLE inference_data_temp AS

    SELECT p.store,
        p.dept,
        p.date,
        p.is_holiday,
        st.type,
        st.size,
        f.temperature,
        f.fuel_price,
        f.factor1,
        f.factor2,
        f.factor3,
        f.factor4,
        f.factor5,
        f.cpi,
        f.unemployment,
        NULL::DOUBLE PRECISION AS weekly_sales
    FROM plan AS p
    LEFT JOIN stores AS st
        ON p.store = st.store
    LEFT JOIN features AS f
        ON p.store = f.store
        AND p.dept = f.dept
        AND p.date = f.date

    UNION ALL

    SELECT
        s.store,
        s.dept,
        s.date,
        s.is_holiday,
        st.type,
        st.size,
        f.temperature,
        f.fuel_price,
        f.factor1,
        f.factor2,
        f.factor3,
        f.factor4,
        f.factor5,
        f.cpi,
        f.unemployment,
        s.weekly_sales::DOUBLE PRECISION AS weekly_sales
    FROM sales AS s
    LEFT JOIN stores AS st
        ON s.store = st.store
    LEFT JOIN features AS f
        ON s.store = f.store
        AND s.dept = f.dept
        AND s.date = f.date;
    """

    hook.run(create_inference_data_temp)

    get_values = """
    SELECT COUNT(*),
    MIN(date),
    MAX(date)
    FROM inference_data_temp
    """
    values = hook.get_first(get_values)

    logging.info(f"""Количество строк в таблице: {values[0]}
Минимальная дата: {values[1]}
Максимальная дата: {values[2]}""")

    min_date = hook.get_first("SELECT min(date) FROM plan")[0]

    ti = context['ti']
    ti.xcom_push(key='first_plan_date', value=min_date)

    


def preprocess_features(**context):

    ti = context['ti']
    first_plan_date = ti.xcom_pull(task_ids='load_data', key='first_plan_date')
    first_plan_date = pd.to_datetime(first_plan_date)

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    sql_query = """
    SELECT * 
    FROM inference_data_temp
    ORDER BY store, dept, date"""

    inference_data_temp = hook.get_pandas_df(sql_query)

    inference_data_temp['date'] = pd.to_datetime(inference_data_temp['date'])

    processed_df = preprocess_data(inference_data_temp)

    inference_df = processed_df[processed_df['date'] >= first_plan_date].copy()

    inference_df = inference_df.drop('weekly_sales', axis=1)

    lag_columns = ["sales_1week_ago",
                   "sales_2week_ago",
                   "sales_4week_ago",
                   "mean_sales_2week",
                   "mean_sales_4week",
                   "avg_sales_before"]
    
    inference_df[lag_columns] = inference_df[lag_columns].fillna(0)


    for_delete = ['temperature',
        'fuel_price',
        'factor1',
        'factor2',
        'factor4',
        'cpi',
        'unemployment',
        'year',
        'date']
    
    model_features = [feature for feature in inference_df.columns if feature not in for_delete] 

    inference_df = inference_df.dropna(
        subset=model_features
    )

    ti.xcom_push(key='inference_df', value=inference_df.to_json())

    delete_table = """DROP TABLE inference_data_temp"""
    hook.run(delete_table)



def load_model_from_s3(**context):

    import tempfile
    from io import BytesIO
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        endpoint_url='https://storage.yandexcloud.net'
    )

    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)

    bio = BytesIO(obj['Body'].read())
    model = pickle.load(bio)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        model_path = tmp_file.name
        pickle.dump(model, tmp_file)

    ti = context['ti']
    ti.xcom_push(key='model_path', value=model_path)



def run_batch_inference(**context):

    ti = context['ti']

    inference_df_json = ti.xcom_pull(task_ids='preprocess_features', key='inference_df')
    model_path = ti.xcom_pull(task_ids='load_model', key='model_path')

    inference_df = pd.read_json(inference_df_json)

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    for_delete = ['temperature',
            'fuel_price',
            'factor1',
            'factor2',
            'factor4',
            'cpi',
            'unemployment',
            'year',
            'date']
    
    model_features = [feature for feature in inference_df.columns if feature not in for_delete] 
    
    predict = model.predict(inference_df[model_features])
    inference_df['predicted_weekly_sales'] = predict

    inference_df.loc[inference_df['predicted_weekly_sales'] < 0, 'predicted_weekly_sales'] = 0

    ti.xcom_push(key='final_df', value=inference_df.to_json())



def save_predictions_to_postgres(**context):

    ti = context['ti']
    predictions_json = ti.xcom_pull(task_ids='run_inference', key='final_df')
    
    predictions_df = pd.read_json(predictions_json)
    

    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
    

    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()


    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        store INT,
        dept INT,
        date DATE,
        predicted_weekly_sales FLOAT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (store, dept, date)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    
    from psycopg2.extras import execute_values
    
    values = predictions_df[['store', 'dept', 'date', 'predicted_weekly_sales']].to_records(index=False).tolist()

    insert_query = """
    INSERT INTO predictions (store, dept, date, predicted_weekly_sales)
    VALUES %s
    ON CONFLICT (store, dept, date) 
    DO UPDATE SET 
        predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
        prediction_timestamp = CURRENT_TIMESTAMP;
    """
    
    execute_values(cursor, insert_query, values)
    conn.commit()

    count_lines = pg_hook.get_first("SELECT COUNT(*) FROM predictions")[0]
    logging.info(f'Финальная таблица: {count_lines} строк')

    lines = pg_hook.get_records("SELECT * FROM predictions LIMIT 5")
    for line in lines:
        logging.info(line)
    cursor.close()
    conn.close()




default_args = {
    'owner': 'Туева Анна Николаевна',
    'depends_on_past': False,
    'email': ['best_data_scientist_in_the_world@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'sales_prediction_batch_inference',
    default_args=default_args,
    description='Batch-инференс прогнозирования продаж для Прилавка',
    schedule_interval='0 20 * * 0', 
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'batch-inference', 'production'],
)

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_postgres,
    provide_context=True,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_features',
    python_callable=preprocess_features,
    provide_context=True,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_s3,
    provide_context=True,
    dag=dag,
)

task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=run_batch_inference,
    provide_context=True,
    dag=dag,
)

task_save_predictions = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions_to_postgres,
    provide_context=True,
    dag=dag,
)

task_load_data >> task_preprocess
task_load_model >> task_inference
task_preprocess >> task_inference >> task_save_predictions
