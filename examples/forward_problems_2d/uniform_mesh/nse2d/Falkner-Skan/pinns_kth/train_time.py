# Code used to time the training process

import numpy as np
from tensorflow.keras import models, layers, optimizers, activations
from matplotlib import pyplot as plt
from time import time
from PINN_FS import PINNs
import wandb
from datetime import datetime

# %%
#################
# DATA loading
#################
d = np.load('FalknerSkan_n0.08.npz')

u = d['u'].T
v = d['v'].T
x = d['x'].T
y = d['y'].T
p = d['p'].T
x = x - x.min()
y = y - y.min()
ref = np.stack((u, v, p))


#################
# Training Parameters
#################
act = "tanh"
n_adam = 1000  # number of iterations
nn = 20
nl = 6
cp_step = 29
bc_step = 10
method = "L-BFGS-B"

# wandb params
now = datetime.now()
dateprefix = now.strftime("%d_%b_%Y_%H_%M")
run_name = "KTH_Pinns_Timing"
wandb.init(
    project="ICCFD_KTH_PINNs",
    entity="starslab-iisc",
    name=run_name,
    config={
        "act": act,
        "n_adam": n_adam,
        "nn": nn,
        "nl": nl,
        "cp_step": cp_step,
        "bc_step": bc_step,
        "method": method,
    },
)


# %%
#################
# Training Data
#################

# Collection points
cp = np.concatenate((x[:, ::cp_step].reshape((-1, 1)), y[:, ::cp_step].reshape((-1, 1))), axis=1)
n_cp = len(cp)

print(f"INFO: Number of collection points = {n_cp}")
print(f"INFO: Shape of Collection points = {cp.shape}")


# Boundary points
ind_bc = np.zeros(x.shape, dtype=bool)
ind_bc[[0, -1], ::bc_step] = True
ind_bc[:, [0, -1]] = True

x_bc = x[ind_bc].flatten()
y_bc = y[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()

bc = np.array([x_bc, y_bc, u_bc, v_bc]).T

ni = 2
nv = bc.shape[1] - ni + 1
pp = 1

# Randomly select half of Boundary points
indx_bc = np.random.choice([False, True], len(bc), p=[1 - pp, pp])
bc = bc[indx_bc]

print("INFO: Number of boundary points = ", len(bc))
print("INFO: Shape of Boundary points = ", bc.shape)

n_bc = len(bc)
test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{n_bc}'

# %%
#################
# Compiling Model
#################

inp = layers.Input(shape=(ni,))
hl = inp
for i in range(nl):
    hl = layers.Dense(nn, activation=act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())
lr = 1e-3
opt = optimizers.Adam(lr)
pinn = PINNs(model, opt, n_adam)

#################
# Training Process
#################
print(f"INFO: Start training case : {test_name}")
st_time = time()

hist = pinn.fit(bc, cp)

en_time = time()
comp_time = en_time - st_time
print(f"INFO: Training time = {comp_time}")

wandb.log({"Training Time": comp_time})
wandb.log({"Training Time per Iteration": comp_time / n_adam})
