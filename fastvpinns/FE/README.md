# Development Notes
---

### Author : Thivin Anandh
### Date   : 30-Aug-2023

---

## 1. Introduction
---

This folder contains all the FEM Basis funcitons for most of the commonly used shape functions. All the basis functions are imported from the base class called `basis_function_2d`, which is defined in the file `basis_function_2d.py`. The basis functions are defined in the following files:

   1. value(xi, eta) - This will return the value of the basis function at the reference point (xi, eta)
   2. gradx(xi, eta) - This will return the value of the derivative of the basis function with respect to xi
   3. grady(xi, eta) - This will return the value of the derivative of the basis function with respect to eta
   4. gradxx(xi, eta) - This will return the value of the second derivative of the basis function with respect to xi
   5. gradxy(xi, eta) - This will return the value of the second derivative of the basis function with respect to xi and eta

## To Add a new Basis Function
---

Add a new python file for the new basis function as described in the existing files and dont forget to add the imports to the end of `basis_function_2d.py` file.


