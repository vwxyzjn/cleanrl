from setuptools import setup, find_packages

setup(name='cleanrl',
      version='0.1.0',
      install_requires=['gym', 'torch', 'tensorboard', 'wandb'],
      packages=find_packages()
)
