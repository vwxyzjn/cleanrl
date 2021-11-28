FROM gitpod/workspace-full:latest
USER gitpod

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN sudo apt-get update && \
    sudo apt-get -y install python3-pip xvfb ffmpeg git build-essential python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
RUN pip install poetry

# install mujoco
RUN sudo apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
