# %%
import numpy as np
import matplotlib.pyplot as plt

d = np.load("FalknerSkan_n0.08.npz")
bc_step = 10
u = d['u'].T
v = d['v'].T
x = d['x'].T
y = d['y'].T
p = d['p'].T
x = x - x.min()
y = y - y.min()
ref = np.stack((u, v, p))
# print(f"Shape of ref = {ref.shape}")

test_points = np.vstack((x.flatten(), y.flatten())).T
exact_solution = ref.reshape((3, -1)).T
# %%
