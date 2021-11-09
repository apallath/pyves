from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("VERSION", "r") as vh:
    version = vh.read()
version = version.strip()

setup(name='ves',
      python_requires='>3.7',
      version=version,
      author='Akash Pallath',
      author_email='apallath@seas.upenn.edu',
      description='Variationally enhanced sampling using PyTorch and OpenMM.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/apallath/ves-torch',
      packages=['ves'])
