FROM nvcr.io/nvidia/pytorch:21.03-py3
ENV NVIDIA_DISABLE_REQUIRE=1
RUN mkdir -p /setup
COPY ./requirements.txt /setup
WORKDIR /setup
RUN pip install -r requirements.txt --no-cache-dir --use-feature=2020-resolver