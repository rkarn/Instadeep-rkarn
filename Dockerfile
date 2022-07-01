FROM jupyter/scipy-notebook

RUN pip install joblib 

WORKDIR /app

ADD . .
