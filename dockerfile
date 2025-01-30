FROM python:3.9-slim

WORKDIR /app

COPY app /app
COPY models /models

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port=8501 --server.enableCORS=false"]