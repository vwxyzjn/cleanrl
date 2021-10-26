FROM vwxyzjn/cleanrl-base:latest

COPY ./cleanrl /cleanrl

RUN poetry lock
RUN poetry install -E atari
RUN poetry install -E pybullet
