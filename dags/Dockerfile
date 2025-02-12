FROM apache/airflow:2.6.3-python3.9

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 전체 홈 디렉토리 생성 및 권한 설정
RUN mkdir -p /home/airflow && \
    mkdir -p /home/airflow/.cache/huggingface && \
    chown -R 50000:0 /home/airflow && \
    chmod -R 775 /home/airflow/.cache

# requirements.txt를 복사할 디렉토리 명시
COPY requirements.txt /opt/airflow/requirements.txt

USER airflow
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" && \
    pip install --no-cache-dir -r /opt/airflow/requirements.txt

WORKDIR /opt/airflow