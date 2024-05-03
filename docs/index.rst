.. fastvpinns documentation master file, created by
   sphinx-quickstart on Mon Apr 29 02:17:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FastVPINNs's documentation!
======================================

.. image:: https://github.com/cmgcds/fastvpinns/actions/workflows/unit-tests.yml/badge.svg
   :alt: Unit tests
   :target: https://github.com/cmgcds/fastvpinns/actions/workflows/unit-tests.yml

.. image:: https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml/badge.svg
   :alt: Integration tests
   :target: https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml

.. image:: https://github.com/cmgcds/fastvpinns/actions/workflows/compatibility-tests.yml/badge.svg
   :alt: Compatibility check
   :target: https://github.com/cmgcds/fastvpinns/actions/workflows/compatibility-tests.yml

.. image:: https://img.shields.io/badge/Coverage-92%25-brightgreen
   :alt: Coverage

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :alt: MIT License
   :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Code style: black
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue
   :alt: Python Versions

A robust tensor-based deep learning framework for solving PDE's using hp-Variational Physics-Informed Neural Networks (hp-VPINNs). The framework supports handling complex geometries and uses tensor-based loss computation to accelerate the training of conventional hp-VPINNs. 
The framework is based on the work by `FastVPINNs Paper <https://arxiv.org/abs/2404.12063>`_. The framework is written on `Tensorflow 2.0 <https://www.tensorflow.org/>`_ and has support for handling external meshes. 

*Note: This framework is an highly optimised version of the the initial implementation of hp-VPINNs by* `kharazmi <https://github.com/ehsankharazmi/hp-VPINNs>`_. Ref `hp-VPINNs(arXiv) <https://arxiv.org/abs/2003.05385>`_.

Variational Physics-informed neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variational Physics-Informed neural networks are special form of physics informed neural networks, which uses variational form of the loss function to train the NN. A special form of hp-Variational PINNs which uses h- & p- refinement to enhance the ability of the NN to capture higher frequency solutions. 
For more details on the theory and implementation of hp-VPINNs, please refer to the `FastVPINNs Paper <https://arxiv.org/abs/2404.12063>`_ and `hp-VPINNs Paper <https://arxiv.org/abs/2003.05385>`_.

.. include an image here
.. image:: images/vpinns.png
   :alt: VPINNs Image


.. toctree::
    :maxdepth: 3
    :caption: Getting Started

    Installation <_rst/_installation.rst>

.. toctree::
    :maxdepth: 3
    :caption: Example-Problems

      Tutorials <_rst/_tutorial.rst>

.. toctree::
      :maxdepth: 2
      :caption: API Reference

      fastvpinns <_rst/fastvpinns.rst>

.. toctree::
      :maxdepth: 1
      :caption: Community

      License <_rst/_licence.rst>
      Contributing <_rst/_contributing>
      Cite FastVPINNs <_rst/_cite.rst>
      Team & Fundings <_rst/_team.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
