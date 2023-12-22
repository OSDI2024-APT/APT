from setuptools import setup, find_packages
import os

setup(name='apt',
      version='0.1',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      data_files=[('apt', ['../build/libapt.so'])])