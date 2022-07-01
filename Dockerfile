FROM jupyter/scipy-notebook

WORKDIR /app

RUN jupyter notebook

ADD run.sh /tmp/run.sh
RUN chmod +x /tmp/run.sh
ENTRYPOINT ["/tmp/run.sh"]

ADD . .
