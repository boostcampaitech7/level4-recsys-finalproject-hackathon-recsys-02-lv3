FROM apache/airflow:2.7.3-python3.9

USER airflow

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt