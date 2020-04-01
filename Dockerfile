FROM ubuntu:16.04
LABEL maintainer="Dimitri K. Sifoua <dimitri.sifoua@gmail.com>"
ARG DEBIAN_FRONTEND=noninteractive

# Install apt packages
RUN apt-get update
RUN apt-get install -yq python3-pip htop nano git wget libglib2.0-0 ffmpeg

# Install python modules
ADD requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Setup jupyter
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8080" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

WORKDIR /root
EXPOSE 8888 7007
CMD ["jupyter", "notebook", "--no-browser", "--allow-root"]
