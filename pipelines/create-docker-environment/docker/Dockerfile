# Use an official Python runtime as a parent image
FROM nvidia/cuda:9.0-base

ARG user_id
ARG user_name
RUN groupadd -r $user_name && useradd -r -u $user_id -g $user_name $user_name
RUN chown $user_name /home

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install -U git+https://github.com/pseeth/audio_embed.git
RUN conda install pytorch torchvision -c pytorch


USER $user_name
ENV JUPYTER_PASSWORD sha1:9043ec94ab4d:021f1afd47a6b5798dca9e1c4387e0207f400263
# Obtain the SHA hash for your chosen password and copy it above. To do this, use:
# from notebook.auth import passwd
# passwd()
# You'll be asked to put in your password twice. The SHA value will display.
# Copy it (without the single quotes) and paste it above. This will be the password you use to login to
# the Jupyter server.

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password='${JUPYTER_PASSWORD}'" >> /home/$user_name/.jupyter/jupyter_notebook_config.py
RUN jupyter nbextension enable --py widgetsnbextension

ENTRYPOINT ["popper"]
