FROM python:3.11.9-slim
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["predict.py","churn-model.bin","./"]

EXPOSE 9696