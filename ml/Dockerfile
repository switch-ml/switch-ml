FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl

RUN apt-get install unzip

RUN apt-get install -y python3-pip python3-dev libgl1-mesa-glx locales locales-all libglib2.0-0 libsm6 libxrender1 libxext6\
    && cd /usr/local/bin \
    && apt-get install tesseract-ocr -y \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

WORKDIR /app

COPY ml/requirements.txt ./

RUN pip3 install -r requirements.txt

COPY ml/. .


EXPOSE 8000

CMD ["python", "app.py"]