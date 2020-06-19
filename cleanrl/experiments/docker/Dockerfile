FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential
RUN pip install gym[box2d,atari] pybullet==2.7.9
RUN git clone https://github.com/vwxyzjn/cleanrl && \
    cd cleanrl && pip install -e .
RUN apt-get -y install python-opengl
RUN pip install opencv-python
RUN pip install seaborn pandas

WORKDIR /workspace/cleanrl/cleanrl

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD python ppo2_continuous_action.py --capture-video --total-timesteps 200
