Docker Installation
===================

The docker image is available on `Docker Hub <https://hub.docker.com/r/thivinanandh/fastvpinns>`_. This docker version is based on a `Ubuntu 20.04` image with `Python 3.10` installed. The Docker image can support GPU acceleration as it comes with `CUDA 11.1` and `cuDNN 8.0` installed. 


Pre-requisite for installing Docker with GPU Support
____________________________________________________

For GPU support, you need to install `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ on your machine. The toolkit is required to run the docker container with GPU support. Follow all the guidelines in installing the necessary drivers and the toolkit for GPU support. For installing docker on your machine, you can follow the instructions provided in the `Docker Documentation <https://docs.docker.com/engine/install/ubuntu/>`_.


Installing FastVPINNs Docker Image
__________________________________

To install the docker image, you can use the following command:

.. code-block:: bash

    docker pull thivinanandh/fastvpinns:latest

Step by Step Guide to Run the Docker Image
__________________________________________

1. Create a new folder to mount the volume for the docker container. This is used to share the files between the host and the container. Lets create a folder named `fastvpinns` in your home directory.

.. code-block:: bash

    mkdir ~/fastvpinns

2. Change the directory to the folder `fastvpinns`:

.. code-block:: bash

    cd ~/fastvpinns

3. Pull the docker image using the following command:

.. code-block:: bash

    docker pull thivinanandh/fastvpinns:latest

4. To run the docker container with GPU support, you can use the following command:

.. code-block:: bash

    docker run --gpus all -it --rm -v ~/fastvpinns:/fastvpinns thivinanandh/fastvpinns:latest

for CPU only support, you can use the following command:

.. code-block:: bash

    docker run -it --rm -v ~/fastvpinns:/fastvpinns thivinanandh/fastvpinns:latest

Explanation of the command:
- `--gpus all` : This flag is used to enable GPU support for the container.
- `-it` : This flag is used to run the container in interactive mode.
- `--rm` : This flag is used to remove the container once the container is stopped.
- `-v ~/fastvpinns:/fastvpinns` : This flag is used to mount the volume `~/fastvpinns` to the container at `/fastvpinns`.
- `thivinanandh/fastvpinns:latest` : This is the docker image name and tag.

5. When loaded into the container, The console will look like the image shown below. Here it shows whether the GPU is detected and the version of the installed packages. 

.. image:: ../images/console.png
   :alt: Docker Image
   :align: center

6. Navigate to the `examples` folder to run the examples provided in the repository. The Status of GPU can be viewed by running the following command:

.. code-block:: bash

    nvidia-smi


Additional Notes
________________

- To make the experience better, you can attach the docker container to a VSCode instance. This can be done by installing the `Remote - Containers <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_ extension in VSCode. This extension allows you to attach the running docker container to a VSCode instance. This way you can edit the files in the container using the VSCode editor.


