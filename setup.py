from setuptools import setup, find_packages

with open('VERSION', 'r') as f:
      version = f.read().strip()

setup(name='many_world',
      packages=find_packages(),
      install_requires=["mujoco-py", "gym"],
      description='Many-world Environment, for Object-centric RL',
      author='Ge Yang',
      url='',
      author_email='ge.ike.yang@gmail.com',
      version=version)
