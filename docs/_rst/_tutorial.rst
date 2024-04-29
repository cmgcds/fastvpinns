FastVPINNs Tutorials
====================

This page contains tutorials for solving 2D Partial Differential Equations (PDEs) using the FastVPINNs library. The tutorials are organized as follows:

Forward Problems
------------------
Problems on Uniform Mesh
~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1
   :titlesonly:

    Poisson Equation on unit square <tutorials/forward_problems_2d/uniform_mesh/poisson_2d/poisson2d_uniform.rst>
    Helmholtz Equation on unit square <tutorials/forward_problems_2d/uniform_mesh/helmholtz_2d/helmholtz2d_uniform.rst>

Problems on Complex geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :titlesonly:

    Poisson Equation on unit circle <tutorials/forward_problems_2d/complex_mesh/poisson2d/poisson2d.rst>
    Helmholtz Equation on unit circle <tutorials/forward_problems_2d/complex_mesh/helmholtz2d/helmholtz2d.rst>
    Convection-Diffusion Equation on spur-gear geometry <tutorials/forward_problems_2d/complex_mesh/cd2d_gear/cd2d_gear.rst>
    Convection-Diffusion Equation on unit circle <tutorials/forward_problems_2d/complex_mesh/cd2d/cd2d.rst>


Problems with Hard boundary constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1
   :titlesonly:

    Poisson Equation with hard boundary constraints on unit square <tutorials/forward_problems_2d/hard_boundary_constraints/poisson_2d/poisson2d_hard.rst>

Inverse Problems
----------------
.. toctree::
   :maxdepth: 1
   :titlesonly:
   
    Inverse problems with constant inverse parameter <tutorials/inverse_problems_2d/const_inverse_poisson2d/inverse_constant.rst>
    Inverse problems with spatially varying inverse parameter <tutorials/inverse_problems_2d/domain_inverse_cd2d/domain_inverse.rst>
