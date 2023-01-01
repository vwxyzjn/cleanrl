FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
RUN pip install poetry
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY README.md README.md
COPY ./cleanrl/ppo_continuous_action_isaacgym /cleanrl/ppo_continuous_action_isaacgym
RUN poetry config virtualenvs.create false
RUN poetry install
RUN poetry install --with envpool,jax
RUN poetry run pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # install mujoco
# RUN apt-get -y install wget unzip software-properties-common \
#     libgl1-mesa-dev \
#     libgl1-mesa-glx \
#     libglew-dev \
#     libosmesa6-dev patchelf
# RUN poetry install --with mujoco
# RUN poetry run python -c "import mujoco_py"

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# copy local files
COPY ./cleanrl /cleanrl
