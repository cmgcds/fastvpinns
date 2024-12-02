# Main Class which will hold the information of all the FE spaces
# of all the cells within the given mesh
# Author: Thivin Anandh D
# Date:  30/Aug/2023

import numpy as np
import meshio
from .FE2D_Cell_supg import FE2D_Cell

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from tqdm import tqdm

# import plotting
import matplotlib.pyplot as plt

#import path
from pathlib import Path

#import tensorflow
import tensorflow as tf

from utils.print_utils import print_table

from pyDOE import lhs
import pandas as pd

from matplotlib import rc
from cycler import cycler
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20

plt.rcParams['legend.fontsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.prop_cycle'] = cycler(color=['darkblue', '#d62728', '#2ca02c', '#ff7f0e', '#bcbd22', '#8c564b', '#17becf', '#9467bd', '#e377c2', '#7f7f7f'])

class Fespace2D:
        
    # constructor
    def __init__(self, mesh, cells, boundary_points, cell_type: str, fe_order: int, fe_type: str, quad_order: int, quad_type: str, fe_transformation_type: str,  \
                    bound_function_dict: dict, bound_condition_dict: dict, forcing_function, output_path: str, generate_mesh_plot: bool) -> None:
        """
        The constructor of the fespace2d class.
        """
        self.mesh = mesh
        self.boundary_points = boundary_points
        self.cells = cells
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type

        self.fe_transformation_type = fe_transformation_type
        
        if cell_type == "triangle":
            self.fe_transformation_type = "affine"
            print("==========================================================================")
            print("-----------------WARNING---------------------------------------------------")
            print("[INFO] : Setting the fe_transformation_type to affine for triangle cells")
            print("==========================================================================")
        
        self.output_path = output_path
        self.bound_function_dict = bound_function_dict
        self.bound_condition_dict = bound_condition_dict
        self.forcing_function = forcing_function
        self.generate_mesh_plot = generate_mesh_plot
        
        # to be calculated in the plot function
        self.total_dofs = 0
        self.total_boundary_dofs = 0

        # to be calculated on get_boundary_data_dirichlet function 
        self.total_dirichlet_dofs = 0
        
        # get the number of cells
        self.n_cells = self.cells.shape[0]
        
        self.fe_cell = []
        
        # Function which assigns the fe_cell for each cell
        self.set_finite_elements()
        
        # generate the plot of the mesh
        if self.generate_mesh_plot:
            self.generate_plot(self.output_path)
        # self.generate_plot(self.output_path)
        
        # Obtain boundary Data
        self.dirichlet_boundary_data = self.generate_dirichlet_boundary_data()



        title  = ["Number of Cells", "Number of Quadrature Points", "Number of Dirichlet Boundary Points", "Quadrature Order", "FE Order", "FE Type", "FE Transformation Type"]
        values = [self.n_cells, self.total_dofs, self.total_dirichlet_dofs, self.quad_order, self.fe_order, self.fe_type, self.fe_transformation_type]
        # print the table
        print_table("FE Space Information", ["Property", "Value"], \
                    title, \
                    values)
 
    
    def set_finite_elements(self) -> None:
        """
        This function will assign the finite elements to each cell.
        """

        progress_bar = tqdm(total=self.n_cells, desc='Fe2D_cell Setup', unit='cells_assembled', bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}", colour="blue", ncols=100)

        dof = 0
        for i in range(self.n_cells):
            self.fe_cell.append(FE2D_Cell(self.cells[i], self.cell_type, self.fe_order, self.fe_type, self.quad_order, self.quad_type, self.fe_transformation_type, self.forcing_function))

            # obtain the shape of the basis function (N_test, N_quad)
            dof += self.fe_cell[i].basis_at_quad.shape[1]

            progress_bar.update(1)
        # print the Shape details of all the matrices from cell 0 using print_table function
        title = ["Shape function Matrix Shape", "Shape function Gradient Matrix Shape", "Jacobian Matrix Shape", "Quadrature Points Shape", "Quadrature Weights Shape", "Quadrature Actual Coordinates Shape", "Forcing Function Shape"]
        values = [self.fe_cell[0].basis_at_quad.shape, self.fe_cell[0].basis_gradx_at_quad.shape, self.fe_cell[0].jacobian.shape,  self.fe_cell[0].quad_xi.shape, self.fe_cell[0].quad_weight.shape, self.fe_cell[0].quad_actual_coordinates.shape, self.fe_cell[0].forcing_at_quad.shape]
        print_table("FE Matrix Shapes", ["Matrix", "Shape"], \
                    title, \
                    values)
        

        # update the total number of dofs
        self.total_dofs = dof
        
        
    def generate_plot(self, output_path) -> None:
        """
        This function will generate a plot of the mesh.
        """
        num_cells = self.n_cells
        total_quad = 0
        marker_list = ['o', '.', ',', 'x', '+', 'P', 's', 'D', 'd', '^', 'v', '<', '>', 'p', 'h', 'H']

        print(f"[INFO] : Generating the plot of the mesh")
        # Plot the mesh
        plt.figure(figsize=(6.4,4.8),dpi=300)
        
        # label flag ( to add the label only once)
        label_set = False
        
        # plot every cell as a quadrilateral
        # loop over all the cells
        for i in range(self.n_cells):
            # get the coordinates of the cell
            x = self.fe_cell[i].cell_coordinates[:,0]
            y = self.fe_cell[i].cell_coordinates[:,1]
            
            # add the first point to the end of the array
            x = np.append(x,x[0])
            y = np.append(y,y[0])

            
            plt.plot(x,y,'k-',linewidth=0.5)
            # plt.plot(x,y,'ro',markersize=1.5)
            
            # plot the quadrature points
            x_quad = self.fe_cell[i].quad_actual_coordinates[:,0]
            y_quad = self.fe_cell[i].quad_actual_coordinates[:,1]
            
            total_quad += x_quad.shape[0]
            
            if not label_set:
                plt.scatter(x_quad,y_quad,marker='x',color='b',s=2,label="Quad Pts")
                label_set = True
            else:
                plt.scatter(x_quad,y_quad,marker='x',color='b',s=2)
                pass
            
        self.total_dofs = total_quad 
        
        
        bound_dof = 0
        # plot the boundary points
        #loop over all the boundary tags
        for i,(bound_id,bound_pts) in enumerate(self.boundary_points.items()):
            # get the coordinates of the boundary points
            x = bound_pts[:,0]
            y = bound_pts[:,1]
            
            # add the first point to the end of the array
            x = np.append(x,x[0])
            y = np.append(y,y[0])
            
            bound_dof += x.shape[0]
            
            plt.scatter(x,y,marker=marker_list[i+1],s=2,label=f"Bd-id : {bound_id}")
            
        self.total_boundary_dofs = bound_dof
        
        plt.legend(bbox_to_anchor=(0.85, 1.02))
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        
        # join the output path and filename
        outfile = Path(output_path) / "mesh.png"
        
        plt.savefig(str(Path(output_path) / "mesh.png"),bbox_inches='tight')
        plt.savefig(str(Path(output_path) / "mesh.svg"),bbox_inches='tight')
        
        # print the total number of quadrature points
        print(f"Plots generated")
        print(f"[INFO] : Total number of cells = {self.n_cells}")
        print(f"[INFO] : Total number of quadrature points = {self.total_dofs}")
        print(f"[INFO] : Total number of boundary points = {self.total_boundary_dofs}")


    def generate_dirichlet_boundary_data(self) -> np.ndarray:
        """
        This function will return the boundary points and their values
        """
        x = []
        y = []
        for i,(bound_id,bound_pts) in enumerate(self.boundary_points.items()):
            # get the coordinates of the boundary points
            for pt in bound_pts:
                pt_new = np.array([pt[0],pt[1]],    dtype=np.float64)
                x.append(pt_new)
                val = np.array(self.bound_function_dict[bound_id](pt[0],pt[1]), dtype=np.float64).reshape(-1,1)
                y.append(val)
        
        print(f"[INFO] : Total number of Dirichlet boundary points = {len(x)}")
        self.total_dirichlet_dofs = len(x)
        print(f"[INFO] : Shape of Dirichlet-X = {np.array(x).shape}")
        print(f"[INFO] : Shape of Y = {np.array(y).shape}")

        return x, y

    def generate_dirichlet_boundary_data_vector(self, component) -> np.ndarray:
        """
        This function will return the boundary points and their values
        """
        x = []
        y = []
        for i,(bound_id,bound_pts) in enumerate(self.boundary_points.items()):
            # get the coordinates of the boundary points
            for pt in bound_pts:
                pt_new = np.array([pt[0],pt[1]],    dtype=np.float64)
                x.append(pt_new)
                val = np.array(self.bound_function_dict[bound_id](pt[0],pt[1])[component], dtype=np.float64).reshape(-1,1)
                y.append(val)
        
        return x, y
    
    # def get_dirichlet_data_for_training_lambda(self,id) -> np.ndarray:
    #     """
    #     This function will return the boundary point and its value
    #     for a specific boundary point( here id is just the index of the boundary point)
    #     """
    #     pt = np.array(self.dirichlet_boundary_data[0][id]).reshape(1,2)
    #     val = np.array(self.dirichlet_boundary_data[1][id]).reshape(1,1)
    #     pt = tf.convert_to_tensor(pt,dtype=tf.float64)
    #     val = tf.convert_to_tensor(val,dtype=tf.float64)
    #     return [pt, val]
    
    def get_shape_function_val(self) -> np.ndarray:
        """
        This function will return the shape function reference values for All cells
        """
        shape_func_val_list = []
        for cell_index in range(self.n_cells):
            shape_func_val_list.append(self.fe_cell[cell_index].basis_at_quad.copy())
        
        return np.array(shape_func_val_list)
        
    
    def get_shape_function_grad_x(self) -> np.ndarray:
        """
        This function will return the shape function reference values for All cells
        """
        shape_func_grad_x_list = []
        for cell_index in range(self.n_cells):
            shape_func_grad_x_list.append(self.fe_cell[cell_index].basis_gradx_at_quad.copy())
        
        return np.array(shape_func_grad_x_list)

    def get_shape_function_grad_y(self) -> np.ndarray:
        """
        This function will return the shape function reference values for All cells
        """
        shape_func_grad_y_list = []
        for cell_index in range(self.n_cells):
            shape_func_grad_y_list.append(self.fe_cell[cell_index].basis_grady_at_quad.copy())
        
        return np.array(shape_func_grad_y_list)

    def get_shape_function_val(self, cell_index) -> np.ndarray:
        """
        This function will return the shape function actual values on a given cell
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        return self.fe_cell[cell_index].basis_at_quad.copy()

        
    
    def get_shape_function_grad_x(self, cell_index) -> np.ndarray:
        """
        This function will return the shape function actual values on a given cell
        """

        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        return self.fe_cell[cell_index].basis_gradx_at_quad.copy()
    
    def get_shape_function_grad_x_ref(self, cell_index) -> np.ndarray:
        """
        This function will return the shape function reference values on a given cell
        """

        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        return self.fe_cell[cell_index].basis_gradx_at_quad_ref.copy()
        

    def get_shape_function_grad_y(self, cell_index) -> np.ndarray:
        """
        This function will return the shape function actual values on a given cell
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        return self.fe_cell[cell_index].basis_grady_at_quad.copy()

    def get_shape_function_grad_y_ref(self, cell_index) -> np.ndarray:
        """
        This function will return the shape function referece values on a given cell
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        return self.fe_cell[cell_index].basis_grady_at_quad_ref.copy()
        
    
    def get_quadrature_actual_coordinates(self, cell_index) -> np.ndarray:
        """
        This function will return the actual coordinates of the quadrature points
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        return self.fe_cell[cell_index].quad_actual_coordinates.copy()


    def get_quadrature_weights(self, cell_index) -> np.ndarray:
        """
        This function will return the quadrature weights for a given cell

        Parameters
        ----------
        cell_index : int
            The index of the cell for which the quadrature weights are needed
        
        Returns
        -------
        np.ndarray
            The quadrature weights for the given cell of dimension (N_Quad_Points, 1)
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")

        return self.fe_cell[cell_index].mult.copy()
    
    def get_forcing_function_values(self, cell_index) -> np.ndarray:
        """
        This function will return the forcing function values at the quadrature points
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        # Changed by Thivin: To assemble the forcing function at the quadrature points here in the fespace
        # so that it can be used to handle multiple dimensions on a vector valud problem

         # get number of shape functions
        n_shape_functions = self.fe_cell[cell_index].basis_function.num_shape_functions
        
        # Loop over all the basis functions and compute the integral
        f_integral = np.zeros((n_shape_functions, 1), dtype=np.float64)

        for i in range(n_shape_functions):
            val = 0
            for q in range(self.fe_cell[cell_index].basis_at_quad.shape[1]):
                x = self.fe_cell[cell_index].quad_actual_coordinates[q, 0]
                y = self.fe_cell[cell_index].quad_actual_coordinates[q, 1]
                # print("f_values[q] = ",f_values[q])

                # the JAcobian and the quadrature weights are pre multiplied to the basis functions
                val +=  ( self.fe_cell[cell_index].basis_at_quad[i, q] ) * self.fe_cell[cell_index].forcing_function(x, y) 
                # print("val = ", val)

            f_integral[i] = val

        self.fe_cell[cell_index].forcing_at_quad = f_integral

        return self.fe_cell[cell_index].forcing_at_quad.copy()

    def real_f_values(self, cell_index):
        return self.fe_cell[cell_index].real_f
    

    def get_forcing_function_values_vector(self, cell_index, component) -> np.ndarray:
        """
        This function will return the forcing function values at the quadrature points
        based on the Component of the RHS Needed, for vector valued problems
        """
        if(cell_index > self.n_cells):
            raise ValueError(f"cell_index should be less than {self.n_cells}")
        
        # get number of shape functions
        n_shape_functions = self.fe_cell[cell_index].basis_function.num_shape_functions
        
        # get the coordinates
        x = self.fe_cell[cell_index].quad_actual_coordinates[:, 0]
        y = self.fe_cell[cell_index].quad_actual_coordinates[:, 1]

        # compute the forcing function values
        f_values = self.fe_cell[cell_index].forcing_function(x, y)[component]

        # compute the integral
        f_integral = np.sum(self.fe_cell[cell_index].basis_at_quad * f_values, axis=1)

        self.fe_cell[cell_index].forcing_at_quad = f_integral.reshape(-1, 1)

        return self.fe_cell[cell_index].forcing_at_quad.copy()
    

    def get_pde_data_for_training_lambda(self, cell_index) -> None:
        """
        This function will return one cell's data for training
        which will be used to generate the tf.data.Dataset object
        """
        
        main_data = []

        # get the quadrature points
        x = self.fe_cell[cell_index].quad_actual_coordinates
        
        # get the jacobian values
        y = self.fe_cell[cell_index].jacobian
        
        # print(f"[INFO] : Shape of x = {x.shape}")
        # print(f"[INFO] : Shape of y = {y.shape}")
        x = tf.convert_to_tensor(x,dtype=tf.float64)
        y = tf.convert_to_tensor(y,dtype=tf.float64)
        # append the data
        return x, y
    
    def get_pde_training_data(self) -> None:
        """
        this funciton will return the entire data for training as a tf.tensor
        """
        main_data_x = []
        main_data_y = []

        for cell_index in range(self.n_cells):
            # get the quadrature points
            x = self.fe_cell[cell_index].quad_actual_coordinates
            
            # get the jacobian values
            y = self.fe_cell[cell_index].jacobian

            # append the data
            main_data_x.append(x)
            main_data_y.append(y)
        
        return tf.convert_to_tensor(main_data_x,dtype=tf.float64), tf.convert_to_tensor(main_data_y,dtype=tf.float64)

    
    def get_forcing_data_for_training(self) -> None:
        """
        This function will return the data for training
        It should return a matrix of size N_Quad_Points x N_cells
        Each FE_2D_Cell object will return a forcing function value at each quadrature point
        in shape (N_Quad_Points, 1),
        we need to stack them together to get a matrix of size (N_Quad_Points, N_cells)
        """
        
        main_data = []

        # loop over all the cells
        for cell_index in range(self.n_cells):
            # get the quadrature points
            x = self.fe_cell[cell_index].forcing_at_quad
            
            # append the data
            main_data.append(x)
        
        
        # return the data as a numpy array
        return np.squeeze(np.array(main_data).T,axis=0)
    


    def get_sensor_data(self, exact_solution, num_points):
        """
        Used in inverse problem, to obtain the sensor data (actual solution) at random points
        TODO :  Currently works only for problems with analytical solution
                Need to implement methodologies to obtain sensor data for problems from a file 

                Not implemented for External or complex meshes. 
        """
        # generate random points within the bounds of the domain
        # get the bounds of the domain
        x_min = np.min(self.mesh.points[:,0])
        x_max = np.max(self.mesh.points[:,0])
        y_min = np.min(self.mesh.points[:,1])
        y_max = np.max(self.mesh.points[:,1])
        eps = 0.1
        # sample n random points within the bounds of the domain
        # Generate points in the unit square

        num_internal_points = int(num_points * 0.9)
        num_boundary_points = num_points - num_internal_points

        points = lhs(2, samples=num_internal_points)

        # Scale points to the desired interval
        x = x_min + points[:, 0] * (x_max - x_min)
        y = y_min + points[:, 1] * (y_max - y_min)

        # keep 10% of the points on the boundary
        num_boundary_edges = len(self.boundary_points.items())

        # divide the boundary points equally among the edges
        num_boundary_points_per_edge = int(num_boundary_points / num_boundary_edges)

        for i,(bound_id,bound_pts) in enumerate(self.boundary_points.items()):
            # get the coordinates of the boundary points
            length = bound_pts.shape[0]

            # get the random indices
            indices = np.random.randint(0,length,num_boundary_points_per_edge)

            # get the coordinates
            x_boundary = bound_pts[indices,0]
            y_boundary = bound_pts[indices,1]

            x = np.append(x,x_boundary)
            y = np.append(y,y_boundary)
            


        # stack them together
        points = np.stack((x,y),axis=1)

        # get the exact solution at the points
        exact_sol = exact_solution(points[:,0], points[:,1])

        # print the shape of the points and the exact solution
        print(f"[INFO] : Number of sensor points = {points.shape[0]}")
        print(f"[INFO] : Shape of sensor points = {points.shape}")

        return points, exact_sol

    
    def get_sensor_data_external(self, exact_sol, num_points, file_name):
        """
        This is used to obtain the sensor data for external file
        The file will be in the format of x,y,exact_sol
        """

        # use pandas to read the file
        df = pd.read_csv(file_name)

        x = df.iloc[:,0].values
        y = df.iloc[:,1].values
        exact_sol = df.iloc[:,2].values

        # now sample num_points from the data
        indices = np.random.randint(0,x.shape[0],num_points)

        x = x[indices]
        y = y[indices]
        exact_sol = exact_sol[indices]

        # stack them together
        points = np.stack((x,y),axis=1)

        return points, exact_sol        
