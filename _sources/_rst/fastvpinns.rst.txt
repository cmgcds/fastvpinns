FastVPINNs Module Documentation
===============================

This section covers the documentation of the FastVPINNs modules. The package is divided into several subpackages, each of which is documented in its own section. The main module documentation is provided below.

- :ref:`FE <FE>`        - Finite Element routines for 2D domains
- :ref:`Geometry <Geometry>`  - Meshing and Geometry routines
- :ref:`Model <Model>`        - Dense Neural Network Model
- :ref:`Physics <Physics>`    - Physics classes for the problem
- :ref:`Data <Data>`          - Data handling routines
- :ref:`Utils <Utils>`        - Utility functions


.. _FE:

FE
-----------

----

This section holds the documentation for the FE module of the FastVPINNs package. The module provides the necessary classes and functions to obtain the finite element test functions and quadrature points for the 2D domain within the FastVPINNs package. This section is broadly classified into 

.. toctree::
   :maxdepth: 2
   :titlesonly:

Finite Element Test Functions
''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Jacobi test functions <library/fe2d/fe2d_jacobi.rst>
      Basis function 2d(Abstract) <library/fe2d/fe2d_basis_function.rst>

.. toctree::
   :maxdepth: 2
   :titlesonly:

Quadrature Functions
''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Quadrature Functions(Abstract) <library/fe2d/fe2d_quadratureformulas.rst>
      Quadrature Functions-2D <library/fe2d/fe2d_quadratureformulas2d.rst>

.. toctree::
   :maxdepth: 2
   :titlesonly:

Finite Element Transformations
''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      FE Transformation(Abstract) <library/fe2d/fe2d_transformation.rst>
      Affine Transformations <library/fe2d/fe2d_affine_transformation.rst>
      Bilinear Transformations <library/fe2d/fe2d_bilinear_transformation.rst>
      

.. toctree::
   :maxdepth: 2
   :titlesonly:

Finite Element Setup
''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:
   
      Fespace2D  <library/fe2d/fe2d_fespace2d.rst>
      FE2DSetupMain <library/fe2d/fe2d_fe2d_setup.rst>
      FE2DCell <library/fe2d/fe2d_fe2d_cell.rst>


.. _Geometry:

Geometry
-----------

----

This section holds the documentation for the the geometry module of the FastVPINNs package. The module provides the necessary classes and functions to obtain the geometry information either from the external mesh file or from the internal mesh generation routines. This supports the generation of VTK and GMSH files for visualization purposes. 

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Geometry(Abstract)<library/geometry/geometry.rst>
      Geometry2D <library/geometry/geometry2d.rst>


.. _Model:

Model
-----------

----

This section holds the documentation for the the Model module of the FastVPINNs package. The module provides the necessary classes and functions to train a Variational physics informed neural network (VPINN) model for the given physics problem. The model contains only the information about the neural network architecture and the training process. The physics information is provided by the user in the form of a physics class. The model class is responsible for training the neural network and providing the necessary prediction functions.

.. toctree::
   :maxdepth: 2
   :titlesonly:

Model Types
''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Dense Model - Forward Problem <library/model/model.rst>
      Dense Model - Forward Problem with hard constraints <library/model/model_hard.rst>
      Dense Model - Inverse Problem Constant Coefficient <library/model/model_inverse.rst>
      Dense Model - Inverse Problem Spatially Varying Coefficient <library/model/model_inverse_domain.rst>

.. _Physics:

Physics
-----------

----

This section holds the documentation for the the Physics module of the FastVPINNs package. The module provides the necessary classes and functions to define the physics of the problem. The physics class contains the information about the governing equations and their loss calculation in the variational form. 

.. toctree::
   :maxdepth: 2
   :titlesonly:

Forward Problems
''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Convection-Diffusion 2D <library/physics/cd2d.rst>
      Helmholtz 2D <library/physics/helmholtz2d.rst>
      Poisson 2D <library/physics/poisson2d.rst>

Inverse Problems (Constant Coefficient)
''''''''''''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Convection-Diffusion-2D - Inverse Problem with Constant Coefficient <library/physics/cd2d_inverse.rst>
      Poission-2D - Inverse Problem with Constant Coefficient <library/physics/poisson2d_inverse.rst>

Inverse Problems (Spatially Varying Coefficient)
''''''''''''''''''''''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Convection-Diffusion-2D - Inverse Problem with Spatially Varying Coefficient <library/physics/cd2d_inverse_domain.rst>


.. _Data:

Data
-----------

----

This section holds the documentation for the the datahandler module of the FastVPINNs package. The module provides the necessary classes and functions to handle the data for the training of the VPINN model. The datahandler class is responsible for generating and assembling the tensors into required format for the training process. 

.. toctree::
   :maxdepth: 2
   :titlesonly:

      DataHandler -  Abstract class for all datahandler routines <library/data/datahandler.rst>
      DataHandler2D -  DataHandler routines for 2D <library/data/datahandler2d.rst>


.. _Utils:

Utils
-----------

----

This section holds the documentation for all the helper functions, which are responsible for plotting and console outputs. 

.. toctree::
   :maxdepth: 2
   :titlesonly:

      Plotting utils <library/utils/plot_utils.rst>
      Printing utils <library/utils/print_utils.rst>
      Compute utils <library/utils/compute_utils.rst>
