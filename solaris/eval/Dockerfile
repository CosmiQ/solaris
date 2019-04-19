FROM python:3.6

MAINTAINER Dlindenbaum <dlindenbaum@iqt.org>


## Install RTRee
RUN apt-get update && apt-get install -y --no-install-recommends \
        libspatialindex-dev && \
        pip install rtree

## Install General Requirements
RUN pip install pandas && pip install git+https://github.com/CosmiQ/cw-eval.git
