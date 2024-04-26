# FastVPINNs
--- 
[![Unit tests](https://github.com/cmgcds/fastvpinns/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/cmgcds/fastvpinns/actions/workflows/unit-tests.yml)

[![Integration tests](https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml)

![pylint]()

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust tensor-based deep learning framework for solving PDE's using hp-Variational Physics-Informed Neural Networks (hp-VPINNs). The framework is based on the work by [FastVPINNs Paper](https://arxiv.org/abs/2404.12063). 

This library is an highly optimised version of the the initial implementation of hp-VPINNs by [kharazmi](https://github.com/ehsankharazmi/hp-VPINNs). Ref [hp-VPINNs Paper](https://arxiv.org/abs/2003.05385).

## Authors
---

- [**Thivin Anandh**](https://github.com/thivinanandh)
- [**Divij Ghose**](https://divijghose.github.io/)
- [**Sashikumaar Ganesan**](https://cds.iisc.ac.in/faculty/sashi)

## Installation
---

To install the package, run the following command:

```bash
pip install fastvpinns
```

## Usage
---

For detailed usage, please refer to the README file in the [examples](examples) directory. 

The package provides a simple API to train and solve PDE using VPINNs. The following code snippet demonstrates how to train a hp-VPINN model for the 2D Poisson equation for a structured grid. We could observe that we can  solve a PDE using fastvpinns using effectively 30 lines of code.

```python
#load the geometry 
domain = Geometry_2D("quadrilateral", "internal", 100, 100, "./")
cells, boundary_points = domain.generate_quad_mesh_internal(x_limits=[0, 1],y_limits=[0, 1],n_cells_x=4, n_cells_y=4, num_boundary_points=400)

# load the FEspace
fespace = Fespace2D(domain.mesh,cells,boundary_points,domain.mesh_type,fe_order=5,fe_type="jacobi",quad_order=5,quad_type="legendre", fe_transformation_type="bilinear",bound_function_dict=bound_function_dict,bound_condition_dict=bound_condition_dict,
forcing_function=rhs,output_path=i_output_path,generate_mesh_plot=True)

# Instantiate Data handler 
datahandler = DataHandler2D(fespace, domain, dtype=tf.float32)

# Instantiate the model with the loss function for the model 
model = DenseModel(layer_dims=[2, 30, 30, 30, 1],learning_rate_dict=0.01,params_dict=params_dict,
        loss_function=pde_loss_poisson,  ## Loss function of poisson2D
        input_tensors_list=[in_tensor, dir_in, dir_out],
        orig_factor_matrices=[datahandler.shape_val_mat_list,datahandler.grad_x_mat_list, datahandler.grad_y_mat_list],
        force_function_list=datahandler.forcing_function_list, tensor_dtype=tf.float32,
        use_attention=i_use_attention, ## Archived (not in use)
        activation=i_activation,
        hessian=False)

# Train the model
for epoch in range(1000):
    model.train_step()
```

Note : Supporting functions which define the actual solution and boundary conditions have to be passed to the main code


## Modules

There are 5 important modules in the package. they are
- Geometry
- FE_2D
- data
- model
- physics

### Geometry
This module contains the classes and functions to generate the mesh and the geometry of the domain. The module can read the quadrilateral mesh from external files or generate a structured quadrilateral mesh internally. This module also features vtk file generation for output visualisation and domain and quadrature point plotting for complex domains. 

### FE_2D

This module contains the classes and functions to generate the finite element basis functions and their derivatives for polynomials such as "jacobi" and "legendre" polynomials. The module also contains numerical quadrature routines such as Gauss Legendre and Gauss Jacobi Legendre polynomials which can be used for numerical integration purposes. It also handles the ability to extract and assign values for dirichlet boundary points on complex domains. It computes the RHS matrices to be used in the loss functionals of the hp-VPINNs model. 

#### data

This module contains routines to convert all the data into the tensorial format, which is required for the residual computation of the model in Variational format. This modules also takes care of the tensor data type requried for tensorflow and also controls the precision of calculations.

### model

This module contains the classes and functions required for training the NN to solve the PDE. All of these are dense models, which are fine tuned for specific problems such as problems with hard constraints , inverse problems and inverse problems. 

### physics

This is the module, which captures the core idea of the package, this contains functions responsible for calculating losses for different PDE's such as Poisson-2D, Helmholtz-2D, CD-2D equations for forward problems and also for some of the inverse problems. 


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.