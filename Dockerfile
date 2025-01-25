FROM apache/airflow:2.7.3-python3.9

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev

USER airflow
RUN pip install --upgrade pip

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt