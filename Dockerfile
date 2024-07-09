# this file is to host the official docker image of fastvpinns with python3 and tensorflow

# Download the base image for CUDA Libraries and cuDNN
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# set up timezone
ENV TZ=Asia/Singapore \
    DEBIAN_FRONTEND=noninteractive

# Run to setup Ascii Image Converter
RUN echo 'deb [trusted=yes] https://apt.fury.io/ascii-image-converter/ /' | tee /etc/apt/sources.list.d/ascii-image-converter.list

# Taken from https://stackoverflow.com/questions/56139706/speeding-up-apt-get-update-to-speed-up-docker-image-builds
# To Speedup image builds, replace the default ubuntu mirror with a faster mirror
RUN sed -i 's/htt[p|ps]:\/\/archive.ubuntu.com\/ubuntu\//mirror:\/\/mirrors.ubuntu.com\/mirrors.txt/g' /etc/apt/sources.list

# Install additional packages and restrict python version to be less than 3.11
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    # libGL Packages
    libgl1-mesa-glx \
    libglu1-mesa \
    libglu1 \
    # ascii-image-converter
    ascii-image-converter

# git clone the official repository of fastvpinns
RUN git clone https://github.com/cmgcds/fastvpinns.git

# open the requirements.txt file and replace the "tensorflow>=2.9.1,<=2.13.0" with "tensorflow>=2.9.1,<=2.11.0"
RUN sed -i 's/tensorflow>=2.9.1,<=2.13.0/tensorflow>=2.9.1,<=2.11.0/g' /fastvpinns/requirements.txt

# used for debugging
# COPY requirements.txt /fastvpinns/requirements.txt 

# Set the working directory to the cloned repository
WORKDIR /fastvpinns

# Set the PYTHONPATH environment variable
RUN pip install .

# add /fastvpinns to the current python path
ENV PYTHONPATH "${PYTHONPATH}:/fastvpinns"

# add this to bashrc
RUN echo "export PYTHONPATH=${PYTHONPATH}:/fastvpinns" >> ~/.bashrc

# set alias for python as python3 in bashrc
RUN echo "alias python=python3" >> ~/.bashrc

# temporarily copy the file
COPY docker_initialise.py /fastvpinns/docker_initialise.py
# Change execution permission
RUN chmod +x /fastvpinns/docker_initialise.py


# set bash as the default command
CMD ["/bin/bash"]


ENTRYPOINT ["bash", "-c", "python3 /fastvpinns/docker_initialise.py && exec /bin/bash"]

