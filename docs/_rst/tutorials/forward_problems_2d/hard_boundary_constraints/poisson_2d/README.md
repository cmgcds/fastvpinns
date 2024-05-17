# Solving forward problems with FastVPINNs : Enforcing hard boundary constraints with an ansatz function.

In this example, we will learn how to use hard boundary constraints using FastVPINNs. In particular, we will solve the 2-dimensional Poisson equation, as shown below, while simultaneously estimating the uniform diffusion parameter $\epsilon$ using synthetically generated sensor data.

$$-\epsilon\Delta u(x) = f(x), \quad \ x \in \Omega = (-1, 1)^2
$$
where
$$
f(x,y) = -2\omega^2\sin{\omega x}\sin{\omega y}
$$

We begin by introducing the various files required to run this example

## Contents
- [Example File - sin_cos.py](#example-file): The boundary conditions, forcing function $f$ and parameters are defined in this file.
    - [Defining boundary values](#defining-boundary-values)
    - [Defining the forcing function](#defining-the-forcing-function)
    - [Defining bilinear parameters](#defining-bilinear-parameters)
- [Input File - Input.yaml](#input-file): The input file is used to define the parameters required for the experiment.
    - [experimentation](#experimentation)
    - [geometry](#geometry)
    - [fe](#fe)
    - [pde](#pde)
    - [model](#model)
    - [logging](#logging)
- [Main File - main_poisson2d_hard.py](#main-file): The main file is used to run the experiment.
    - [Import relevant FastVPINNs methods](#import-relevant-fastvpinns-methods)
    - [Reading the Input File](#reading-the-input-file)
    - [Setting up a `Geometry_2D` object](#setting-up-a-geometry_2d-object)
    - [Reading the boundary conditions and values](#reading-the-boundary-conditions-and-values)
    - [Setting up the finite element space](#setting-up-the-finite-element-space)
    - [Defining the hard boundary constraint ansatz](#defining-the-hard-boundary-constraint-ansatz)
    - [Instantiating a model with hard boundary constraints](#instantiating-a-model-with-hard-boundary-constraints)
- [Training the model](#training-the-model)

The code in this example can be run using
```bash
python3 main_poisson2d_hard.py input.yaml
```

## Example File
The example file, `sin_cos.py`, defines the boundary conditions and boundary values, the forcing function and exact function (if test error needs to be calculated), bilinear parameters and the actual value of the parameter that needs to be estimated (if the error between the actual and estimated parameter needs to be calculated) 
### Defining boundary values
Since this example ecforces zero Dirichlet boundary conditions using hard constraints, the boundary functions defined in the example file are not used. Instead, the ansatz function for hard boundary constraints is defined in the [main file](#main-file)

### Defining the forcing function
`rhs` can be used to define the forcing function $f$.
```python
def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # f_temp =  32 * (x  * (1 - x) + y * (1 - y))
    # f_temp = 1

    omegaX = 4.0 * np.pi
    omegaY = 4.0 * np.pi
    f_temp = -2.0 * (omegaX**2) * (np.sin(omegaX * x) * np.sin(omegaY * y))

    return f_temp
```

### Defining bilinear parameters
The bilinear parameters like diffusion constant can be defined by `get_bilinear_params_dict`
```python
def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 1.0

    return {"eps": eps}
```
Here, `eps` denoted the diffusion constant.



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



[Back to contents](#contents)

## Main file
This is the main file which needs to be run for the experiment, with the input file as an argument. For the example, we will use the main file `main_poisson2d_hard.py`

Following are the key components of a FastVPINNs main file 

### Import relevant FastVPINNs methods

```python
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.Geometry.geometry_2d import Geometry_2D
```
Will import the functions related to setting up the finite element space, 2D Geometry and the datahandler required to manage data and make it available to the model.

```python
from fastvpinns.model.model_hard import DenseModel_Hard
``` 
Will import the model file where the neural network and its training function is defined. The model file `model_hard.py` contains the `DenseModel_Hard` class. The `call` function in this model applies the hard boundary constraint function to the output of the neural network, and the `train_step` function does not add a supervised boundary loss to the PDE residual for training.

```python
from fastvpinns.physics.poisson2d import pde_loss_poisson
```
Imports the loss function for the 2-dimensional Poisson problem.
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

### Defining the hard boundary constraint ansatz

The ansatz function for applying zero Dirichlet hard boundary contraints can be defined using `apply_hard_boundary_constraints`
```python
    @tf.function
    def apply_hard_boundary_constraints(inputs, x):
        """This method applies hard boundary constraints to the model.
        :param inputs: Input tensor
        :type inputs: tf.Tensor
        :param x: Output tensor from the model
        :type x: tf.Tensor
        :return: Output tensor with hard boundary constraints
        :rtype: tf.Tensor
        """
        ansatz = (
            tf.tanh(4.0 * np.pi * inputs[:, 0:1])
            * tf.tanh(4.0 * np.pi * inputs[:, 1:2])
            * tf.tanh(4.0 * np.pi * (inputs[:, 0:1] - 1.0))
            * tf.tanh(4.0 * np.pi * (inputs[:, 1:2] - 1.0))
        )
        ansatz = tf.cast(ansatz, i_dtype)
        return ansatz * x
```
Here, the ansatz we use is of the form $\tanh{(4\pi x)}\times\tanh{(4\pi(x-1))}\times\tanh{(4\pi y)}\times\tanh{(4\pi(y-1))}$

### Instantiating a model with hard boundary constraints

```python
    model = DenseModel_Hard(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
        hard_constraint_function=apply_hard_boundary_constraints,
    )
```
`DenseModel_Hard` is a model written for inverse problems with spatially varying parameter estimation. In this problem, we pass the loss function `pde_loss_poisson` from the `physics` file `poisson2d.py`.


### Training the model
We are now ready to train the model to approximate the solution of the PDE. 

```python
for epoch in range(num_epochs):

        # Train the model
        batch_start_time = time.time()

        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
        ...
```

[Back to contents](#contents)

## Solution
---
<div style="display: flex; justify-content: space-around;">
    <figure>
        <img src="exact_solution.png" alt="Exact Solution">
        <figcaption style="text-align: center;">Exact Solution</figcaption>
    </figure>
    <figure>
        <img src="predicted_solution.png" alt="Predicted Solution">
        <figcaption style="text-align: center;">Predicted Solution</figcaption>
    </figure>
    <figure>
        <img src="error.png" alt="Error">
        <figcaption style="text-align: center;">Error</figcaption>
    </figure>
</div>

## References
---

1. [FastVPINNs: Tensor-Driven Acceleration of VPINNs for Complex Geometries.](https://arxiv.org/abs/2404.12063)
