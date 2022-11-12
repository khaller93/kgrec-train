FROM python:3.9-buster

LABEL maintainer="Kevin Haller <contact@kevinhaller.dev>"
LABEL description="Application that enables experiments with KG-based recommendation. It provides a command line interface to train models."

RUN adduser --disabled-password --gecos '' ohm

RUN mkdir /var/lib/kgrec \
    && chown ohm:ohm -R /var/lib/kgrec

USER ohm

WORKDIR /var/lib/kgrec

COPY . /opt/kgrec-app

RUN cd /opt/kgrec-app \
    && pip install -r requirements.txt

LABEL version="1.0.0"

ENTRYPOINT ["python", "/opt/kgrec-app/main.py"]