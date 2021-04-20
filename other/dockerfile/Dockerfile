FROM ubuntu
RUN sed -i 's/archive.ubuntu.com/mirrors.tencentyun.com/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 npm python3-pip git \
    && echo "python3 install success" \
    && npm i jsdom -g \
    && mkdir /data \
    && mkdir /data/stock \
    && cd /data/stock
#    && git clone