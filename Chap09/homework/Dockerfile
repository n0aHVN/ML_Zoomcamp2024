FROM agrigorev/model-2024-hairstyle:v3

RUN pip install numpy==1.23.1
RUN pip install tensorflow==2.14.0


COPY homework.py .
ENV MODEL_NAME=model_2024_hairstyle_v2.tflite
CMD [ "homework.lambda_handler" ]
