# %%

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

act = "tanh"
n_adam = 1000
n_neural = 20
n_layer = 8
cp_step = 1
bc_step = 1
method = "L-BFGS-B"


def get_dirichlet_and_test_data_external(filename, dtype, num_bound_points_per_edge):
    """
    Function is a custom function to get boundary and test data from the external file
    """
    d = pd.read_csv(filename, delimiter=',', dtype=np.float64)
    bc_step = 2
    # Column values
    # ['x', 'y', 'u1', 'u2', 'p', ' u1_grad_x', ' u1_grad_y', ' u2_grad_x',
    # ' u2_grad_y', ' vorticity']
    x = d['x'].values.reshape(1000, 80)
    y = d['y'].values.reshape(1000, 80)
    u = d['u1'].values.reshape(1000, 80)
    v = d['u2'].values.reshape(1000, 80)
    p = d['p'].values.reshape(1000, 80)
    u_x = d[' u1_grad_x'].values.reshape(1000, 80)
    u_y = d[' u1_grad_y'].values.reshape(1000, 80)
    v_x = d[' u2_grad_x'].values.reshape(1000, 80)
    v_y = d[' u2_grad_y'].values.reshape(1000, 80)
    vorticity = d[' vorticity'].values.reshape(1000, 80)

    ref = np.vstack((u, v, p))
    # print(f"Shape of ref = {ref.shape}")

    test_points = np.vstack((x.flatten(), y.flatten())).T
    exact_solution = ref.reshape((3, -1)).T

    # convert to tensor
    test_points = tf.constant(test_points, dtype=dtype)
    exact_solution = tf.constant(exact_solution, dtype=dtype)

    # print(f"Shape of x = {x.shape}")
    ind_bc = np.zeros(x.shape, dtype=bool)
    ind_bc[[0, -1],] = True
    ind_bc[::bc_step, [0, -1]] = True

    x_bc = x[ind_bc].flatten()
    y_bc = y[ind_bc].flatten()

    print(
        f"Shape of x_bc = {x_bc.shape} :-> x = 0 {x_bc[x_bc == 0].shape} :-> x = 20 {x_bc[x_bc == 20].shape} :-> x = 40 {x_bc[x_bc == 40].shape} :-> x = 60 {x_bc[x_bc == 60].shape} :-> x = 80 {x_bc[x_bc == 80].shape} "
    )
    print(
        f"Shape of y_bc = {y_bc.shape} :-> y = -0.5 {y_bc[y_bc == -0.5].shape} :-> y = 0.5 {y_bc[y_bc == 0.5].shape} "
    )

    # input bc
    input_dirichlet = np.hstack((x_bc[:, None], y_bc[:, None]))

    u_bc = u[ind_bc].flatten()
    v_bc = v[ind_bc].flatten()

    # output dirichlet
    output_dirichlet_u = u_bc[:, None]
    output_dirichlet_v = v_bc[:, None]

    # print(f"Shape of input_dirichlet = {input_dirichlet.shape}")
    # print(f"Shape of output_dirichlet_u = {output_dirichlet_u.shape}")
    # print(f"Shape of output_dirichlet_v = {output_dirichlet_v.shape}")

    dirichlet_dict_input = {}
    dirichlet_dict_output = {}

    # assign value for each component
    dirichlet_dict_input[0] = tf.constant(input_dirichlet, dtype=dtype)
    dirichlet_dict_output[0] = tf.constant(output_dirichlet_u, dtype=dtype)

    dirichlet_dict_input[1] = tf.constant(input_dirichlet, dtype=dtype)
    dirichlet_dict_output[1] = tf.constant(output_dirichlet_v, dtype=dtype)

    # print(f"Shape of cp = {cp.shape}")
    cmp = sns.color_palette('RdBu_r', as_cmap=True)
    for title, data in zip(["u", "v", "p"], [u, v, p]):
        plt.figure()
        plt.set_cmap(cmp)
        plt.contourf(x[:, : 50 * 15], y[:, : 50 * 15], data[:, : 50 * 15], cmap='RdBu_r', levels=50)
        # plot the color bar in horizontal orientation
        plt.colorbar(orientation='horizontal', pad=0.1, aspect=50)
        # custom aspect ratio of length to be 10 and height to be 2
        plt.gca().set_aspect(3, adjustable='box')
        plt.title(f"{title}")
        plt.show()

    return dirichlet_dict_input, dirichlet_dict_output, test_points, exact_solution, x.shape


a, b, c, d, e = get_dirichlet_and_test_data_external(
    'RE_NR_100.000000_Backward_Step.csv', tf.float32, 100
)
# %%
