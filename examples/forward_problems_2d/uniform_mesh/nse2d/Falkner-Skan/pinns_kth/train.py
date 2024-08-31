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
n_adam = 10000  # number of iterations
nn = 20
nl = 6
cp_step = 29
bc_step = 10
method = "L-BFGS-B"

# wandb params
now = datetime.now()
dateprefix = now.strftime("%d_%b_%Y_%H_%M")
run_name = "KTH_Pinns_Ablation" + "_" + dateprefix
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

# upload the current file to wandb
wandb.save('train.py')
wandb.save('PINN_FS.py')

# %%
#################
# Prediction
#################
cpp = np.array([x.flatten(), y.flatten()]).T

pred = pinn.predict(cpp)
u_p = pred[:, 0].reshape(u.shape)
v_p = pred[:, 1].reshape(u.shape)
p_p = pred[:, 2].reshape(u.shape)

# Shift the pressure to the reference level before calculating the error
# Becacuse we only have pressure gradients in N-S eqs but not pressure itself in BC
deltap = p.mean() - p_p.mean()
p_p = p_p + deltap
pred = np.stack((u_p, v_p, p_p))


pred = np.stack((u_p, v_p, p_p))
ref = np.stack((u, v, p))
names = ["U", "V", "P"]
# %%
for i, name in enumerate(names):
    fig, axs = plt.subplots(3, figsize=(9, 10))
    axs[0].contourf(x, y, pred[i, :, :])
    axs[1].contourf(x, y, ref[i, :, :])
    clb = axs[2].contourf(x, y, np.abs(ref[i, :, :] - pred[i, :, :]))
    cbar = plt.colorbar(clb, orientation="horizontal", pad=0.3)
    axs[0].set_title(f"{name} Prediction", fontdict={"size": 18})
    axs[1].set_title("Reference", fontdict={"size": 18})
    axs[2].set_title("Error", fontdict={"size": 18})
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"pinns_{name}.png", dpi=300)

err = np.stack((np.abs(u - u_p), np.abs(v - v_p), np.abs(p - p_p)))

# compute the l1 error, l2 error, l_inf error and their relative errors
# flatten the arrays
u = u.flatten()
v = v.flatten()
p = p.flatten()
u_p = u_p.flatten()
v_p = v_p.flatten()
p_p = p_p.flatten()


l1_error_u = np.mean(np.abs(u - u_p))
l2_error_u = np.sqrt(np.mean(np.square(u - u_p)))
l_inf_error_u = np.max(np.abs(u - u_p))
rel_l1_error_u = l1_error_u / np.mean(np.abs(u))
rel_l2_error_u = l2_error_u / np.sqrt(np.mean(np.square(u)))
rel_l_inf_error_u = l_inf_error_u / np.max(np.abs(u))

l1_error_v = np.mean(np.abs(v - v_p))
l2_error_v = np.sqrt(np.mean(np.square(v - v_p)))
l_inf_error_v = np.max(np.abs(v - v_p))
rel_l1_error_v = l1_error_v / np.mean(np.abs(v))
rel_l2_error_v = l2_error_v / np.sqrt(np.mean(np.square(v)))
rel_l_inf_error_v = l_inf_error_v / np.max(np.abs(v))


l1_error_p = np.mean(np.abs(p - p_p))
l2_error_p = np.sqrt(np.mean(np.square(p - p_p)))
l_inf_error_p = np.max(np.abs(p - p_p))
rel_l1_error_p = l1_error_p / np.mean(np.abs(p))
rel_l2_error_p = l2_error_p / np.sqrt(np.mean(np.square(p)))
rel_l_inf_error_p = l_inf_error_p / np.max(np.abs(p))

print(f"INFO: l1_error_u = {l1_error_u}")
print(f"INFO: l2_error_u = {l2_error_u}")
print(f"INFO: l_inf_error_u = {l_inf_error_u}")
print(f"INFO: Relative l1_error_u = {rel_l1_error_u}")
print(f"INFO: Relative l2_error_u = {rel_l2_error_u}")
print(f"INFO: Relative l_inf_error_u = {rel_l_inf_error_u}")

print(f"INFO: l1_error_v = {l1_error_v}")
print(f"INFO: l2_error_v = {l2_error_v}")
print(f"INFO: l_inf_error_v = {l_inf_error_v}")
print(f"INFO: Relative l1_error_v = {rel_l1_error_v}")
print(f"INFO: Relative l2_error_v = {rel_l2_error_v}")
print(f"INFO: Relative l_inf_error_v = {rel_l_inf_error_v}")

print(f"INFO: l1_error_p = {l1_error_p}")
print(f"INFO: l2_error_p = {l2_error_p}")
print(f"INFO: l_inf_error_p = {l_inf_error_p}")
print(f"INFO: Relative l1_error_p = {rel_l1_error_p}")
print(f"INFO: Relative l2_error_p = {rel_l2_error_p}")
print(f"INFO: Relative l_inf_error_p = {rel_l_inf_error_p}")

# log all values to wandb
wandb.log({"l1_error_u": l1_error_u, "l2_error_u": l2_error_u, "l_inf_error_u": l_inf_error_u})
wandb.log({"l1_error_v": l1_error_v, "l2_error_v": l2_error_v, "l_inf_error_v": l_inf_error_v})
wandb.log({"l1_error_p": l1_error_p, "l2_error_p": l2_error_p, "l_inf_error_p": l_inf_error_p})
wandb.log(
    {
        "rel_l1_error_u": rel_l1_error_u,
        "rel_l2_error_u": rel_l2_error_u,
        "rel_l_inf_error_u": rel_l_inf_error_u,
    }
)
wandb.log(
    {
        "rel_l1_error_v": rel_l1_error_v,
        "rel_l2_error_v": rel_l2_error_v,
        "rel_l_inf_error_v": rel_l_inf_error_v,
    }
)
wandb.log(
    {
        "rel_l1_error_p": rel_l1_error_p,
        "rel_l2_error_p": rel_l2_error_p,
        "rel_l_inf_error_p": rel_l_inf_error_p,
    }
)

# save the prediction as a numpy file
np.savez_compressed(
    'Pinns_kth_FS' + test_name, pred=pred, ref=ref, x=x, y=y, hist=hist, err=err, ct=comp_time
)

# %%
#################
# Save prediction and Model
#################
# np.savez_compressed('pred/res_FS' + test_name, pred = pred, ref = ref, x = x, y = y, hist = hist, err = err, ct = comp_time)
# model.save('models/model_FS' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")
