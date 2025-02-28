import os
from airflow.models.variable import Variable

class Config:
    MONGO_DB_NAME = Variable.get("MONGO_DB_NAME")
    MONGO_DB_PASSWORD = Variable.get("MONGO_DB_PASSWORD")

class Directory:
    AIRFLOW_HOME = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    LIGHTGCN_DIR = os.path.join(AIRFLOW_HOME, 'LightGCN')