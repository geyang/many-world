from setuptools import setup, find_packages

setup(name='many_world',
      packages=find_packages(),
      install_requires=["mujoco-py", "gym"],
      description='Many-world Environment, for Object-centric RL',
      author='Ge Yang',
      url='',
      author_email='ge.ike.yang@gmail.com',
      version='0.0.0')
