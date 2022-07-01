FROM python:3.8.13-bullseye as venv-image

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_FOLDER=/app
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade --quiet pip setuptools && \
        pip install -r ./requirements.txt

WORKDIR $APP_FOLDER

COPY . .

RUN pip install .


FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 as run-image

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Append the current directory to your python path
ENV HOME_DIRECTORY=/app PYTHONPATH=$HOME_DIRECTORY:$PYTHONPATH
# Add Python bin to PATH
ENV PATH=/opt/venv/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3=3.8* && \
    rm -rf /var/lib/apt/lists/* \
    ln -s /usr/bin/python3 /usr/local/bin/python

COPY --from=venv-image /opt/venv/. /opt/venv/

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Copy files into HOME_DIRECTORY
COPY . .
