FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential python-opengl

RUN pip install poetry
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install
CMD ["python", "cleanrl/ppo.py"]