# Solving Inverse Problems with FastVPINNs : Estimation of uniform diffusion parameter on a quadrilateral geometry.

In this example, we will learn how to solve inverse problems using FastVPINNs. In particular, we will solve the 2-dimensional Poisson equation, as shown below, while simultaneously estimating the uniform diffusion parameter $\epsilon$ using synthetically generated sensor data.

$$-\epsilon\Delta u(x) = f(x), \quad \ x \in \Omega = (-1, 1)^2
$$
for the actual solution $u(x, y) = 10 \sin(x) \tanh(x) e^{-\epsilon x^2}$
In this problem, the actual value of the diffusion parameter, $\epsilon_{\text{actual}}$ is 0.3, and we start with an initial guess of $\epsilon_{\text{initial}}=2.0$.

We begin by introducing the various files required to run this example

## Contents
- [Example File](#example-file): The boundary conditions, forcing function $f$ and parameters are defined in this file.
- [Input File](#input_file): The input file contains parameters for the finite element space and neural networks that can be tuned.
- [Main File](#main-file): The main file is the file that is actually run. 

The code in this example can be run using
```bash
python3 main_inverse.py input_inverse.yaml
```

## Example File
The example file, `inverse_uniform.py`, defines the boundary conditions and boundary values, the forcing function and exact function (if test error needs to be calculated), bilinear parameters and the actual value of the parameter that needs to be estimated (if the error between the actual and estimated parameter needs to be calculated) 
### Defining boundary values
The current version of FastVPINNs only implements Dirichlet boundary conditions. The boundary values can be set defining a function for each boundary, 
```python
EPS = 0.3


def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    return val
```
Here `EPS` is the actual value of the diffusion parameter to be estimated. 
In the above snippet, we define a function `left_boundary` which returns the Dirichlet values to be enforced at that boundary. Similarly, we can define more boundary functions like `right_boundary`, `top_boundary` and `bottom_boundary`. Once these functions are defined, we can assign them to the respective boundaries using `get_boundary_function_dict`
```python
def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: bottom_boundary, 1001: right_boundary, 1002: top_boundary, 1003: left_boundary}
```
Here, `1000`, `1001`, etc. are the boundary identifiers obtained from the geometry. Thus, each boundary gets mapped to it boundary value in the dictionary.

### Defining boundary conditions
As explained above, each boundary has an identifier. The function `get_bound_cond_dict` maps the boundary identifier to the boundary condition (only Dirichlet boundary condition is implemented at this point).
```python
def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}
```

### Defining the forcing function
`rhs` can be used to define the forcing function $f$.
```python
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
```

### Defining bilinear parameters
The bilinear parameters like diffusion constant and convective velocity can be defined by `get_bilinear_params_dict`
```python
def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    # Initial Guess
    eps = EPS

    return {"eps": eps}
```
Here, `eps` denoted the diffusion constant.

### Defining the target parameter values for testing
To test if our solver converges to the correct value of the parameter to be estimated, we use the function `get_inverse_params_actual_dict`. 
```python
def get_inverse_params_actual_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = EPS

    return {"eps": eps}
```
This can then be used to calculate some error metric that assesses the performance of our solver.

[Back to Contents](#contents)

## Input file
The input file, `input_inverse.yaml`, is used to define inputs to your solver. These will usually parameters that will changed often throughout your experimentation, hence it is best practice to pass these parameters externally. 
The input file is divided based on the modules which use the parameter in question, as follows - 
### `experimentation`
This contains `output_path`, a string which specifies which folder will be used to store your outputs.

### `geometry`
This section defines the geometrical parameters for your domain. 
1. In this example, we set the `mesh_generation_method` as `"internal"`. This generates a regular quadrilateral domain with a uniform mesh.
2. The  parameters in `internal_mesh_params` define the x and y limits of the quadrilateral domain(`xmin`, `xmax`, `ymin` and `ymax`), number of cells in the domain in the x and y direction (`n_cells_x` and `n_cells_y`), number of total boundary points (`n_boundary_points`) and number of test points in x and y direction (`n_test_points_x` and `n_test_points_y`).
3. `mesh_type` : FastVPINNs currently provides support for quadrilateral elements only.
4. `external_mesh_params` can be used to specify parameters for the external mesh, and can be ignored for this example

### `fe`
The parameters related to the finite element space are defined here.
1. `fe_order` sets the order of the finite element test functions.
2. `fe_type` set which type of polynomial will be used as the finite element test function.
3. `quad_order` is the number of quadrature in each direction in each cell. Thus the total number of quadrature points in each cell will be `quad_order`$^2$
4. `quad_type` specifies the quadrature rule to be used.

### `pde`
`beta` specifies the weight by which the boundary loss will be multiplied before being added to the PDE loss.

### `model`
The parameters pertaining to the neural network are specified here.
1. `model_architecture` is used to specify the dimensions of the neural network. In this example, [2, 30, 30, 30, 1] corresponds to a neural network with 2 inputs (for a 2-dimensional problem), 1 output (for a scalar problem) and 3 hidden layers with 30 neurons each.
2. `activation` specifies the activation function to be used.
3. `use_attention` specifies if attnention layers are to be used in the model. This feature is currently under development and hence should be set to `false` for now.
4. `epochs` is the number of iterations for which the network must be trained.
5. `dtype` specifies which datatype (`float32` or `float64`) will be used for the tensor calculations.
6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
7. `learning_rate` sets the learning rate `initial_learning_rate` if a constant learning rate is used. A learning rate scheduler can be used by toggling `use_lr_scheduler` to True and setting the corresponding decay parameters below it. 

### `logging` 
It specifies the frequency with which the progress bar and console output will be updated, and at what interval will inference be carried out to print the solution image in the output folder.

### `inverse`
Specific inputs only for inverse problems. `num_sensor_points` specifies the number of points in the domain at which the solution is known (or "sensed"). 

[Back to contents](#contents)

## Main file
This is the main file which needs to be run for the experiment, with the input file as an argument. For the example, we will use the main file `main_inverse.py`

Following are the key components of a FastVPINNs main file 

### Import relevant FastVPINNs methods

```python
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.Geometry.geometry_2d import Geometry_2D
```
Will import the functions related to setting up the finite element space, 2D Geometry and the datahandler required to manage data and make it available to the model.

```python
from fastvpinns.model.model_inverse import DenseModel_Inverse
``` 
Will import the model file where the neural network and its training function is defined. The model file `model_inverse.py` contains the `DenseModel_Inverse` class specifically designed for inverse problems where a spatially varying parameter has to be estimated along with the solution.

```python
from fastvpinns.physics.poisson2d_inverse import *
```
Imports the loss function specifically designed for this problem, with a sensor loss added to the PDE and boundary losses.

```python
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
from fastvpinns.utils.print_utils import print_table
```
Imports functions to calculate the loss, plot the results and print outputs to the console.

### Reading the Input File
The input file is loaded into `config` and the input parameters are read and assigned to their respective variables.

### Setting up a `Geometry_2D` object
```python
domain = Geometry_2D(i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path)
```
will instantiate a `Geometry_2D` object, `domain`, with the mesh type, mesh generation method and test points. In our example, the mesh generation method is `internal`, so the cells and boundary points will be obtained using the `generate_quad_mesh_internal` method.
```python
        cells, boundary_points = domain.generate_quad_mesh_internal(
            x_limits=[i_x_min, i_x_max],
            y_limits=[i_y_min, i_y_max],
            n_cells_x=i_n_cells_x,
            n_cells_y=i_n_cells_y,
            num_boundary_points=i_n_boundary_points,
        )
```

### Reading the boundary conditions and values
As explained in [the example file section](#example-file), the boundary conditions and values are read as a dictionary from the example file
```python
bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()
```

### Setting up the finite element space
```python
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
```
`fespace` will contain all the information about the finite element space, including those read from the [input file](#input-file)    

### Instantiating an inverse problem model

```python
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
```
`DenseModel_Inverse` is a model written for inverse problems with spatially varying parameter estimation. In this problem, we pass the loss function `pde_loss_poisson_inverse` from the `physics` file `poisson_inverse.py`.

We are now ready to train the model to approximate the solution of the PDE while estimating the unknown diffusion parameter using the sensor data. 

```python
for epoch in range(num_epochs):

        # Train the model
        batch_start_time = time.time()

        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
        ...
```

[Back to contents](#contents)