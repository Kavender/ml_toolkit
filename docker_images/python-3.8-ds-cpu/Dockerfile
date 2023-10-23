FROM python:3.8-slim-buster

RUN apt-get update && apt-get -y upgrade && apt-get --no-install-recommends -y install build-essential openssh-client git ca-certificates apt-transport-https gnupg2 curl

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

# INSTALL ANY NEEDED PACKAGES
RUN pip install --upgrade pip

EXPOSE 8080

CMD ["python"]


