FROM ubuntu:18.04

LABEL maintainer="Dimitri K. Sifoua <dimitri.sifoua@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

# Install apt packages
RUN apt-get update
RUN apt-get install wget -y
RUN apt-get install git -y

# Install Ananconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b
RUN rm Anaconda3-2020.02-Linux-x86_64.sh
ENV PATH="/root/anaconda3/bin:$PATH"

# Install Python dependencies
# ADD ./requirements.txt ./
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# Setting Jupyter Notebook configurations
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8080" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py

# Expose port for Jupyter
EXPOSE 8080

# Launch Jupyter Notebook
WORKDIR /root
CMD ["jupyter", "notebook", "--no-browser", "--allow-root"]

