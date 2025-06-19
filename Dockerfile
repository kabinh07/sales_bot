FROM python:3.9

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app/main.py"]