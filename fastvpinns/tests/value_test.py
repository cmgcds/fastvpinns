# %%
import numpy as np
import matplotlib.pyplot as plt


x0, y0 = 0, 0
x1, y1 = 0.5, 0
x2, y2 = 0.5, 0.5
x3, y3 = 0, 0.5

x0, y0 = 0.5, 0.5
x1, y1 = 0.5, 1.0
x2, y2 = 0, 1
x3, y3 = 0, 0.5

# x0, y0 = 0.5, 0.5
# x1, y1 = 0, 0.5
# x2, y2 = 0, 0
# x3, y3 = 0.5, 0

# Set cell
xc0 = (x1 + x3) * 0.5
xc1 = (x1 - x0) * 0.5
xc2 = (x3 - x0) * 0.5

yc0 = (y1 + y3) * 0.5
yc1 = (y1 - y0) * 0.5
yc2 = (y3 - y0) * 0.5

detjk = xc1 * yc2 - xc2 * yc1
rec_detjk = 1 / detjk


print("xc0 = ", xc0)
print("xc1 = ", xc1)
print("xc2 = ", xc2)
print("yc0 = ", yc0)
print("yc1 = ", yc1)
print("yc2 = ", yc2)
print("detjk = ", detjk)
print("rec_detjk = ", rec_detjk)


## reference derivatives
values = np.zeros(4)


def gradx_ref(xi, eta):
    values = np.zeros((4, 4))
    values[0] = 0.25 * (-1 + eta)
    values[1] = 0.25 * (1 - eta)
    values[2] = 0.25 * (-1 - eta)
    values[3] = 0.25 * (1 + eta)
    return values


def grady_ref(xi, eta):
    values = np.zeros((4, 4))
    values[0] = 0.25 * (-1 + xi)
    values[1] = 0.25 * (-1 - xi)
    values[2] = 0.25 * (1 - xi)
    values[3] = 0.25 * (1 + xi)
    return values


w = np.array([1, 1, 1, 1])
xi = np.array([-0.577350, 0.577350, -0.577350, 0.577350])
eta = np.array([-0.577350, -0.577350, 0.577350, 0.577350])

grad_x_ref_val = gradx_ref(xi, eta)
grad_y_ref_val = grady_ref(xi, eta)

print("Grad x ref val = \n", grad_x_ref_val)
print("Grad y ref val = \n", grad_y_ref_val)


orig_grad_x = np.zeros_like(grad_x_ref_val)
orig_grad_y = np.zeros_like(grad_y_ref_val)


for i in range(4):  ## loop over N-Test
    grad_x_ref_test = grad_x_ref_val[i]
    grad_y_ref_test = grad_y_ref_val[i]

    for q in range(4):
        orig_grad_x[i, q] = (
            yc2 * grad_x_ref_test[q] - yc1 * grad_y_ref_test[q]
        ) * rec_detjk

        orig_grad_y[i, q] = (
            -xc2 * grad_x_ref_test[q] + xc1 * grad_y_ref_test[q]
        ) * rec_detjk

print("Orig grad x = \n", orig_grad_x)
print("Orig grad y = \n", orig_grad_y)

orig_grad_x = orig_grad_x * detjk
orig_grad_y = orig_grad_y * detjk

print("Orig grad x = \n", orig_grad_x)
print("Orig grad y = \n", orig_grad_y)

print("Sum of grad x = ", np.sum(orig_grad_x))
print("Sum of grad y = ", np.sum(orig_grad_y))

# %%
