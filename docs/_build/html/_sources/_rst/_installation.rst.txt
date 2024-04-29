Installation
============

The build of the code is currently tested on Python 3.8, 3.9 and 3.10. The code is tested on Ubuntu 20.04 and Ubuntu 22.04, 
refer to `Compatibility-CI <https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml>`_ page for more details.

Setting up an Virtual Environment
_________________________________

It is recommended to create a virtual environment to install the package. You can create a virtual environment using the following command:

.. code-block:: bash

    $ python3 -m venv venv
    $ source venv/bin/activate

For conda users, you can create a virtual environment using the following command:

.. code-block:: bash

    $ conda create -n fastvpinns python=3.8
    $ conda activate fastvpinns


Installing via PIP
__________________

You can simply install the package using pip as follows:

.. code-block:: bash

    $ pip install fastvpinns

On ubuntu/mac systems with libGL issues caused due to matplotlib or gmsh, please run the following command to install the required dependencies.

.. code-block:: bash

    $ sudo apt-get install libgl1-mesa-glx


Installing from source
______________________
The official distribution is on GitHub, and you can clone the repository using

.. code-block:: bash
    
    $ git clone https://github.com/cmgcds/fastvpinns.git

To install the package just type:
 
.. code-block:: bash

    $ pip install -e .
