from setuptools import setup

setup(name='blob_env',
      version='0.1.0',
      install_requires=[
          'gym'],
      packages=[
          'blob_env',
          'blob_env.envs',
      ],  # And any other dependencies foo needs
      )
