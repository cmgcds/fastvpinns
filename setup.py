from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fastvpinns",
    version="1.0.0",
    author="Thivin Anandh, Divij Ghose, Sashikumaar Ganesan",
    author_email="thivinanandh@gmail.com",
    description="A short description of your package",
    packages=find_packages(),
    install_requires=requirements,
    keywords=['variational PINNs', 'VPINNs', 'PINNs', 'Physics-informed neural networks', 'Deep learning', 'Machine learning']
)
