FROM ubuntu:latest
COPY ./pip.conf /tmp/
RUN sed -i 's/archive.ubuntu.com/mirrors.tencentyun.com/g' /etc/apt/sources.list \
    && apt-get update \
    && mkdir ~/.pip \
    && cp /tmp/pip.conf ~/.pip/pip.conf
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 npm python3-pip git \
    && echo "python3 install success" \
    && npm i puppeteer -g \
    && npm i readline -g \
    && mkdir /data \
    && mkdir /data/stock \
    && cd /data/stock \
    && pip3 install Pillow \
    && git clone
# docker build  -t simcse:0.3 /data/project/learn_code/docker