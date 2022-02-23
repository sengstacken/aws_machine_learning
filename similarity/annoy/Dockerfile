FROM python:3.7

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN pip3 install --no-cache numpy pandas
RUN pip3 install --no-cache simhash
RUN conda install -c conda-forge python-annoy
#RUN pip3 install --no-cache annoy
RUN pip3 install --no-cache flask

COPY annoy-train.py /usr/bin/train
COPY annoy-serve.py /usr/bin/serve

RUN chmod 755 /usr/bin/train /usr/bin/serve

EXPOSE 8080