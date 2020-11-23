from setuptools import setup

setup(name='blob_env',
      version='0.1.0',
      install_requires=[
          'gym'],
      packages=[
          'envs',
      ],  # And any other dependencies foo needs
      )
