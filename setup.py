from setuptools import setup, find_packages

setup(name='many_world',
      packages=find_packages(),
      install_requires=["mujoco-py", "gym"],
      description='collections of RL environments',
      author='Ge Yang',
      url='',
      author_email='ge.yang@berkeley.edu',
      version='0.0.0')
