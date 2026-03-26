FROM python:3.10-slim        ← start with a clean Python installation

WORKDIR /app                 ← create a folder called /app to work in

COPY requirements.txt .      ← bring in your requirements file
RUN pip install -r ...       ← install all your Python packages

COPY main.py .               ← bring in your actual service code

ENV PORT=8080                ← tell the app to listen on port 8080
EXPOSE 8080

CMD ["uvicorn", "main:app"...] ← start the FastAPI server
