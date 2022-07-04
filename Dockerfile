FROM registry.kao.instadeep.io/library/nvidia/tensorflow:21.02-tf2-py3

WORKDIR /app

RUN apt-get update && apt install cmake libgl1-mesa-dev -y

ADD requirements.txt .
RUN pip install -r requirements.txt
ADD reduced_asnm_nbpo_tasks.pkl .
ADD task_dataset.pkl .
ADD . .
