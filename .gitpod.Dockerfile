FROM gitpod/workspace-full-vnc:latest
USER gitpod
RUN if ! grep -q "export PIP_USER=no" "$HOME/.bashrc"; then printf '%s\n' "export PIP_USER=no" >> "$HOME/.bashrc"; fi

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN sudo apt-get update && \
    sudo apt-get -y install xvfb ffmpeg git build-essential python-opengl

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
RUN pip install poetry --upgrade
RUN poetry config virtualenvs.in-project true

# install mujoco_py
RUN sudo apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
