Solving Inverse Problems with FastVPINNs : Estimation of uniform diffusion parameter on a quadrilateral geometry.
=================================================================================================================

In this example, we will learn how to solve inverse problems using
FastVPINNs. In particular, we will solve the 2-dimensional Poisson
equation, as shown below, while simultaneously estimating the uniform
diffusion parameter :math:`\varepsilon` using synthetically generated
sensor data.

.. math::

   -\varepsilon\Delta u(x) = f(x), \quad \ x \in \Omega = (-1, 1)^2

for the actual solution
:math:`u(x, y) = 10 \sin(x) \tanh(x) e^{-\varepsilon x^2}` In this problem,
the actual value of the diffusion parameter,
:math:`\epsilon_{\text{actual}}` is 0.3, and we start with an initial
guess of :math:`\epsilon_{\text{initial}}=2.0`.

We begin by introducing the various files required to run this example

Contents
--------

-  `Steps to run the code <#steps-to-run-the-code>`__

-  `Example File <#example-file>`__

   -  `Defining the boundary values <#defining-the-boundary-values>`__
   -  `Defining the boundary conditions <#defining-the-boundary-conditions>`__
   -  `Defining the forcing function <#defining-the-forcing-function>`__
   -  `Defining the bilinear parameters <#defining-the-bilinear-parameters>`__
   -  `Defining the target parameter values for testing <#defining-the-target-parameter-values-for-testing>`__

-  `Input File <#input-file>`__

   -  `Experimentation parameters <#experimentation-parameters>`__
   -  `Geometry parameters <#geometry-parameters>`__
   -  `Finite element space parameters <#finite-element-space-parameters>`__
   -  `PDE Beta parameters <#pde-beta-parameters>`__
   -  `Model parameters <#model-parameters>`__
   -  `Logging parameters <#logging-parameters>`__
   -  `Inverse <#inverse>`__

-  `Main File <#main-file>`__

   -  `Import relevant FastVPINNs methods <#import-relevant-fastvpinns-methods>`__
   -  `Reading the Input File <#reading-the-input-file>`__
   -  `Setting up the Geometry2D object <#setting-up-the-geometry2d-object>`__
   -  `Reading the boundary conditions and values <#reading-the-boundary-conditions-and-values>`__
   -  `Setting up the finite element space <#setting-up-the-finite-element-space>`__
   -  `Instantiating the inverse model <#instantiating-the-inverse-model>`__
   -  `Training the model <#training-the-model>`__

-  `Solution <#solution>`__
-  `References <#references>`__

Steps to run the code
---------------------

To run the code, execute the following command:

.. code:: bash

   python3 main_inverse.py input_inverse.yaml

`Back to Contents <#contents>`__

Example File
------------

The example file, ``inverse_uniform.py``, defines the boundary
conditions and boundary values, the forcing function and exact function
(if test error needs to be calculated), bilinear parameters and the
actual value of the parameter that needs to be estimated (if the error
between the actual and estimated parameter needs to be calculated)

`Back to Contents <#contents>`__

Defining the boundary values 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current version of FastVPINNs only
implements Dirichlet boundary conditions. The boundary values can be set
by defining a function for each boundary,

.. code:: python

   EPS = 0.3


   def left_boundary(x, y):
       """
       This function will return the boundary value for given component of a boundary
       """
       val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
       return val

Here ``EPS`` is the actual value of the diffusion parameter to be
estimated. In the above snippet, we define a function ``left_boundary``
which returns the Dirichlet values to be enforced at that boundary.
Similarly, we can define more boundary functions like
``right_boundary``, ``top_boundary`` and ``bottom_boundary``. Once these
functions are defined, we can assign them to the respective boundaries
using ``get_boundary_function_dict``

.. figure:: rect.png
   :alt: Unit Square
   :align: center

.. code:: python

   def get_boundary_function_dict():
       """
       This function will return a dictionary of boundary functions
       """
       return {1000: bottom_boundary, 1001: right_boundary, 1002: top_boundary, 1003: left_boundary}

Here, ``1000``, ``1001``, etc. are the boundary identifiers obtained
from the geometry. Thus, each boundary gets mapped to it boundary value
in the dictionary.

`Back to Contents <#contents>`__

Defining the boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained above, each boundary has an identifier. The function
``get_bound_cond_dict`` maps the boundary identifier to the boundary
condition (only Dirichlet boundary condition is implemented at this
point).

.. code:: python

   def get_bound_cond_dict():
       """
       This function will return a dictionary of boundary conditions
       """
       return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}

`Back to Contents <#contents>`__

Defining the forcing function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``rhs`` function can be used to define the forcing function :math:`f`.

.. code:: python

   def rhs(x, y):
       """
       This function will return the value of the rhs at a given point
       """

       X = x
       Y = y
       eps = EPS

       return (
           -EPS
           * (
               40.0 * X * eps * (np.tanh(X) ** 2 - 1) * np.sin(X)
               - 40.0 * X * eps * np.cos(X) * np.tanh(X)
               + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X)
               + 20 * (np.tanh(X) ** 2 - 1) * np.sin(X) * np.tanh(X)
               - 20 * (np.tanh(X) ** 2 - 1) * np.cos(X)
               - 10 * np.sin(X) * np.tanh(X)
           )
           * np.exp(-1.0 * X**2 * eps)
       )

`Back to Contents <#contents>`__

Defining the bilinear parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bilinear parameters like diffusion constant and convective velocity
can be defined by ``get_bilinear_params_dict``

.. code:: python

   def get_bilinear_params_dict():
       """
       This function will return a dictionary of bilinear parameters
       """
       # Initial Guess
       eps = EPS

       return {"eps": eps}

Here, ``eps`` denoted the diffusion constant.

`Back to Contents <#contents>`__

Defining the target parameter values for testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test if our solver converges to the correct value of the parameter to
be estimated, we use the function ``get_inverse_params_actual_dict``.

.. code:: python

   def get_inverse_params_actual_dict():
       """
       This function will return a dictionary of inverse parameters
       """
       # Initial Guess
       eps = EPS

       return {"eps": eps}

This can then be used to calculate some error metric that assesses the
performance of our solver.

`Back to Contents <#contents>`__

Input file
----------

The input file, ``input_inverse.yaml``, is used to define inputs to your
solver. These will usually parameters that will changed often throughout
your experimentation, hence it is best practice to pass these parameters
externally. The input file is divided based on the modules.

`Back to Contents <#contents>`__

Experimentation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the output path where the results will be saved.

.. code:: yaml

   experimentation:
     output_path: "output/inv_test"  # Path to the output directory where the results will be saved.

`Back to Contents <#contents>`__

Geometry parameters
~~~~~~~~~~~~~~~~~~~

-  In this example, we set the ``mesh_generation_method`` as ``"internal"``. 
   This generates a regular quadrilateral domain with a uniform mesh. 
-  The parameters in ``internal_mesh_params`` define the x and y limits of
   the quadrilateral domain(\ ``xmin``, ``xmax``, ``ymin`` and ``ymax``),
   number of cells in the domain in the x and y direction (``n_cells_x``
   and ``n_cells_y``), number of total boundary points
   (``n_boundary_points``) and number of test points in x and y direction
   (``n_test_points_x`` and ``n_test_points_y``).
-  ``mesh_type`` is set to “quadrilateral” which means that the mesh is
   a quadrilateral mesh. Note: As of now, only quadrilateral meshes are
   supported. So, ``mesh_type`` is set to quadrilateral.
-  ``boundary_sampling_method`` is set to “uniform” which means that the
   boundary points are sampled using the “uniform” method. (Use only
   uniform sampling as of now.)
-  ``external_mesh_params`` can be used to specify parameters for the
   external mesh, and can be ignored for this example.

.. code:: yaml

   geometry:
      mesh_generation_method: "internal"
      internal_mesh_params:
         x_min: -1
         x_max: 1
         y_min: -1
         y_max: 1
         n_cells_x: 2
         n_cells_y: 2
         n_boundary_points: 2000
         n_test_points_x: 100
         n_test_points_y: 100

      mesh_type: "quadrilateral"
      external_mesh_params:
         mesh_file_name: "meshes/rect_quad.mesh"  # should be a .mesh file
         boundary_refinement_level: 8
         boundary_sampling_method: "uniform"  # "uniform" 
   
`Back to Contents <#contents>`__

Finite element space parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameters related to the finite element space are defined here.
-  ``fe_order`` sets the order of the finite element test functions.

-  ``fe_type`` set which type of polynomial will be used as the finite element test function.

- ``quad_order`` is the number of quadrature in each direction in each cell. Thus the total number of quadrature points in each cell will be ``quad_order``\ :math:`^2`

-  ``quad_type`` specifies the quadrature rule to be used.

.. code:: yaml

   fe:
      fe_order: 10    
      fe_type: "jacobi"   #"parmoon", "legendre" and "legendre".
      quad_order:  40    
      quad_type: "gauss-jacobi"  # "gauss-jacobi, gauss-legendre, gauss-lobatto"

`Back to Contents <#contents>`__

PDE beta parameters
~~~~~~~~~~~~~~~~~~~

The ``beta`` specifies the weight by which the boundary loss will be
multiplied before being added to the PDE loss.

.. code:: yaml

   pde:
     beta: 10  # Parameter for the PDE.

`Back to Contents <#contents>`__

Model parameters
~~~~~~~~~~~~~~~~

The parameters pertaining to the neural network are specified here.

-  ``model_architecture`` is used to specify the dimensions of the neural network. In this example, [2, 30, 30, 30, 1] corresponds to a neural network with 2 inputs (for a 2-dimensional problem), 1 output (for a scalar problem) and 3 hidden layers with 30 neurons each.
-  ``activation`` specifies the activation function to be used.
-  ``use_attention`` specifies if attention layers are to be used in the model. This feature is currently under development and hence should be set to ``false`` for now.
-  ``epochs`` is the number of iterations for which the network must be trained.
-  ``dtype`` specifies which datatype (``float32`` or ``float64``) will be used for the tensor calculations.
-  ``set_memory_growth``, when set to ``True`` will enable tensorflow’s memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to ``False`` for now.
-  ``learning_rate`` sets the learning rate ``initial_learning_rate`` if a constant learning rate is used.
-  A learning rate scheduler can be used by toggling ``use_lr_scheduler`` to True and setting the corresponding decay parameters below it. The ``decay_steps`` parameter is the number of steps between each learning rate decay. The ``decay_rate`` parameter is the decay rate for the learning rate. The ``staircase`` parameter is a flag indicating whether to use the staircase decay.

.. code:: yaml

   model:
      model_architecture: [2, 30,30,30, 1]
      activation: "tanh"
      use_attention: False
      epochs: 10000
      dtype: "float32"
      set_memory_growth: False
      learning_rate:
         initial_learning_rate: 0.001
         use_lr_scheduler: False
         decay_steps: 1000
         decay_rate: 0.9
         staircase: False

`Back to Contents <#contents>`__

Logging parameters
~~~~~~~~~~~~~~~~~~

It specifies the frequency with which the progress bar and console
output will be updated, and at what interval will inference be carried
out to print the solution image in the output folder.

.. code:: yaml

   logging:
     update_console_output: 5000  # Number of steps between each update of the console output.

`Back to Contents <#contents>`__

Inverse
~~~~~~~

Specific inputs only for inverse problems. ``num_sensor_points``
specifies the number of points in the domain at which the solution is
known (or “sensed”).

.. code:: yaml

   inverse:
      num_sensor_points: 50

`Back to Contents <#contents>`__

Main file
---------

This is the main file which needs to be run for the experiment, with the
input file as an argument. For the example, we will use the main file
``main_inverse.py``

Following are the key components of a FastVPINNs main file

`Back to Contents <#contents>`__

Import relevant FastVPINNs methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from fastvpinns.data.datahandler2d import DataHandler2D
   from fastvpinns.FE.fespace2d import Fespace2D
   from fastvpinns.Geometry.geometry_2d import Geometry_2D

Will import the functions related to setting up the finite element
space, 2D Geometry and the datahandler required to manage data and make
it available to the model.

.. code:: python

   from fastvpinns.model.model_inverse import DenseModel_Inverse

Will import the model file where the neural network and its training
function is defined. The model file ``model_inverse.py`` contains the
``DenseModel_Inverse`` class specifically designed for inverse problems
where a spatially varying parameter has to be estimated along with the
solution.

.. code:: python

   from fastvpinns.physics.poisson2d_inverse import *

Will import the loss function specifically designed for this problem, with a
sensor loss added to the PDE and boundary losses.

.. code:: python

   from fastvpinns.utils.compute_utils import compute_errors_combined
   from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
   from fastvpinns.utils.print_utils import print_table

Will import functions to calculate the loss, plot the results and print
outputs to the console.

`Back to Contents <#contents>`__

Reading the Input File
~~~~~~~~~~~~~~~~~~~~~~

The input file is loaded into ``config`` and the input parameters are
read and assigned to their respective variables.

`Back to Contents <#contents>`__

Setting up the Geometry2D object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   domain = Geometry_2D(i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path)

will instantiate a ``Geometry_2D`` object, ``domain``, with the mesh
type, mesh generation method and test points. In our example, the mesh
generation method is ``internal``, so the cells and boundary points will
be obtained using the ``generate_quad_mesh_internal`` method.

.. code:: python

           cells, boundary_points = domain.generate_quad_mesh_internal(
               x_limits=[i_x_min, i_x_max],
               y_limits=[i_y_min, i_y_max],
               n_cells_x=i_n_cells_x,
               n_cells_y=i_n_cells_y,
               num_boundary_points=i_n_boundary_points,
           )

`Back to Contents <#contents>`__

Reading the boundary conditions and values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in `the example file section <#example-file>`__, the
boundary conditions and values are read as a dictionary from the example
file

.. code:: python

   bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

`Back to Contents <#contents>`__

Setting up the finite element space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

       fespace = Fespace2D(
           mesh=domain.mesh,
           cells=cells,
           boundary_points=boundary_points,
           cell_type=domain.mesh_type,
           fe_order=i_fe_order,
           fe_type=i_fe_type,
           quad_order=i_quad_order,
           quad_type=i_quad_type,
           fe_transformation_type="bilinear",
           bound_function_dict=bound_function_dict,
           bound_condition_dict=bound_condition_dict,
           forcing_function=rhs,
           output_path=i_output_path,
       )

``fespace`` will contain all the information about the finite element
space, including those read from the `input file <#input-file>`__

`Back to Contents <#contents>`__

Instantiating the inverse model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

       model = DenseModel_Inverse(
           layer_dims=i_model_architecture,
           learning_rate_dict=i_learning_rate_dict,
           params_dict=params_dict,
           loss_function=pde_loss_poisson_inverse,
           input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
           orig_factor_matrices=[
               datahandler.shape_val_mat_list,
               datahandler.grad_x_mat_list,
               datahandler.grad_y_mat_list,
           ],
           force_function_list=datahandler.forcing_function_list,
           sensor_list=[points, sensor_values],
           inverse_params_dict=inverse_params_dict,
           tensor_dtype=i_dtype,
           use_attention=i_use_attention,
           activation=i_activation,
           hessian=False,
       )

``DenseModel_Inverse`` is a model written for inverse problems with
spatially varying parameter estimation. In this problem, we pass the
loss function ``pde_loss_poisson_inverse`` from the ``physics`` file
``poisson_inverse.py``.

`Back to Contents <#contents>`__

Training the model
~~~~~~~~~~~~~~~~~~

We are now ready to train the model to approximate the solution of the
PDE while estimating the unknown diffusion parameter using the sensor
data.

.. code:: python

   for epoch in range(num_epochs):

           # Train the model
           batch_start_time = time.time()

           loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
           ...

`Back to Contents <#contents>`__

Solution
--------

      .. figure:: exact_solution.png
         :alt: Exact Solution
         :align: center

         Exact Solution

      .. figure:: predicted_solution.png
         :alt: Predicted Solution
         :align: center

         Predicted Solution

      .. figure:: error.png
         :alt: Error
         :align: center

         Error

      .. figure:: inverse_eps_prediction.png
         :alt: inverse_eps_prediction
         :align: center

         inverse_eps_prediction

      .. figure:: loss_function.png
         :alt: Train Loss
         :align: center

         Train Loss

`Back to Contents <#contents>`__

References
-----------

1. `FastVPINNs: Tensor-Driven Acceleration of VPINNs for Complex
   Geometries. <https://arxiv.org/abs/2404.12063>`__

`Back to Contents <#contents>`__