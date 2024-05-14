from setuptools import setup, find_packages
import os

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


# Moved all the metadata to the pyproject.toml file
setup(
    name="fastvpinns",
    # version="1.0.5",
    # author="Thivin Anandh, Divij Ghose, Sashikumaar Ganesan",
    # author_email="thivinanandh@gmail.com",
    # description="An Efficient tensor-based implementation of hp-Variational Physics-informed Neural Networks (VPINNs) for solving PDEs",
    packages=find_packages(),
    # long_description=read_file('README.md'),
    # home_page="https://cmgcds.github.io/fastvpinns",
    # license="MIT",
    install_requires=requirements,
    # keywords=['variational PINNs', 'VPINNs', 'PINNs', 'Physics-informed neural networks', 'Deep learning', 'Machine learning']
)
