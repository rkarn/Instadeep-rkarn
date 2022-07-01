FROM jupyter/scipy-notebook

WORKDIR /app

ADD run.sh /tmp/run.sh
RUN chmod +x /tmp/run.sh
ENTRYPOINT ["/tmp/run.sh"]

ADD . .
