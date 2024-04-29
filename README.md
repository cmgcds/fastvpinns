# FastVPINNs - Fast hp-Variational Physics-Informed Neural Networks for solving PDE's
--- 
[![Unit tests](https://github.com/cmgcds/fastvpinns/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/cmgcds/fastvpinns/actions/workflows/unit-tests.yml)
[![Integration tests](https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml)
[![Compatability check](https://github.com/cmgcds/fastvpinns/actions/workflows/compatibility-tests.yml/badge.svg)](https://github.com/cmgcds/fastvpinns/actions/workflows/compatibility-tests.yml)
![Coverage](https://img.shields.io/badge/Coverage-92%25-brightgreen)

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python Versions](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue)

<img src="Fastvpinns_logo.png" alt="Fastvpinns Logo" style="display: block; margin-left: auto; margin-right: auto; width: 30%">


A robust tensor-based deep learning framework for solving PDE's using hp-Variational Physics-Informed Neural Networks (hp-VPINNs). The framework is based on the work by [FastVPINNs Paper](https://arxiv.org/abs/2404.12063).

This library is an highly optimised version of the the initial implementation of hp-VPINNs by [kharazmi](https://github.com/ehsankharazmi/hp-VPINNs). Ref [hp-VPINNs Paper](https://arxiv.org/abs/2003.05385).

## Authors
---

[Thivin Anandh](https://github.com/thivinanandh), [Divij Ghose](https://divijghose.github.io/), [Sashikumaar Ganesan](https://cds.iisc.ac.in/faculty/sashi)

STARS Lab, Department of Computational and Data Sciences, Indian Institute of Science, Bangalore, India

## Installation
---

The build of the code is currently tested on Python 3.8, 3.9 and 3.10. The code is tested on Ubuntu 20.04 and Ubuntu 22.04, Macos-latest and Windows-latest(refer to [Compatibility-CI](https://github.com/cmgcds/fastvpinns/actions/workflows/integration-tests.yml) page for more details).

You can simply install the package using pip as follows:

```bash
pip install fastvpinns
```

 On ubuntu systems with libGL issues caused due to matplotlib or gmsh, please run the following command to install the required dependencies.
```bash
sudo apt-get install -y libglu1-mesa 
```

## Usage
---

For detailed usage, please refer to the README file in the [examples](examples) directory. 

The package provides a simple API to train and solve PDE using VPINNs. The following code snippet demonstrates how to train a hp-VPINN model for the 2D Poisson equation for a structured grid. We could observe that we can  solve a PDE using fastvpinns using effectively 15 lines of code.

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

## Contributing
---
This code is currently maintained by the [Authors](#Authors). We welcome contributions from the community. Please create a pull request with the changes you would like to make.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 