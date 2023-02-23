FROM nvcr.io/nvidia/pytorch:22.12-py3

COPY requirements.txt /opt/app/requirements.txt
COPY subnetworks /opt/app/subnetworks
WORKDIR /opt/app
RUN python3 -m pip install -r requirements.txt

#COPY . /opt/app

CMD "ls"
