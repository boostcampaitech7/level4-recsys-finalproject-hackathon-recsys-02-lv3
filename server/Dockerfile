# 베이스 이미지 설정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /server  

# 필요한 파일 복사
COPY requirements.txt ./
COPY . .  

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 환경변수 설정
ENV PYTHONUNBUFFERED=1

# 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "app/config/log_config.yaml"]
