FROM python:3.8.1-slim-buster

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

RUN apt-get update
RUN apt-get install libgomp1
RUN pip install --upgrade pip
RUN pip install -U jupyter jupyter_contrib_nbextensions
RUN jupyter nbextension enable --py widgetsnbextension

EXPOSE 8080

ENV NAME BasePythonDockerImage

CMD ["jupyter", "notebook", "--ip='*'", "--port=8080", "--no-browser", "--allow-root"]
