# SageMaker TF Inference image
FROM google/tf-serving-scann

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY code /opt/ml/code
COPY savedscann /opt/ml/model/index
WORKDIR /opt/ml/code
ENTRYPOINT ["python","serve"]
