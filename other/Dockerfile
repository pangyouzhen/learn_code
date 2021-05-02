FROM ubuntu
RUN sed -i 's/archive.ubuntu.com/mirrors.tencentyun.com/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 npm python3-pip git \
    && echo "python3 install success" \
    && npm i puppeteer -g \
    && npm i readline -g \
    && mkdir /data \
    && mkdir /data/stock \
    && cd /data/stock \
    && pip3 install Pillow \
    && pip3 install torch \
    && pip3 install torchvision \
    && pip3 install loguru
#    && git clone