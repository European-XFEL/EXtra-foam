FROM ubuntu:16.04

MAINTAINER Jun <zhujun981661@gmail.com>

RUN apt-get update \
    && apt-get install -y build-essential git \
    && apt-get install -y libglib2.0 \
    && apt-get install -y python3 python3-pip python3-pyqt5

RUN git clone https://github.com/European-XFEL/karabo-bridge-py.git \
    && cd karabo-bridge-py \
    && pip3 install .

RUN git clone https://github.com/European-XFEL/karabo_data.git \
    && cd karabo_data \
    && pip3 install .

RUN pip3 install pyFAI

COPY ./ ./fxe-tools

RUN cd fxe-tools && pip3 install -e .
