from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name="afmmech",
    version="1.0",
    packages=find_packages(),
    author="Marshall R. McCraw",
    author_email="marshall.mccraw@yale.edu",
    description="a few simple python scripts for extracting mechanical properties from afm data",
    url="https://github.com/mmccraw98/afmmech",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)