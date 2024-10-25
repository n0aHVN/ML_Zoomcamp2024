FROM svizor/zoomcamp-model:3.11.5-slim
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["dv.bin","model1.bin","q4.py","./"]

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "q4:app"]