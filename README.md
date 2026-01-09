# Generalized PDE Solver Using PINN Neural Networks

The present study builds upon the findings in open literature by implementing a physics-inspired neural network and systematically comparing its accuracy to the finite element method (FEM), which serves as a benchmark for evaluation. While this work does not claim to introduce novel contributions, it aims to highlight the disparities between traditional numerical methods and machine learning-based ones. The key objectives met with the study are as follows:

- The development of a generalized physics-inspired neural network capable of solving PDEs of varying orders.
- A comparative study evaluating the accuracy, computational complexity, and efficiency of PINNs against the finite difference method.

## Repository Structure

- **fdm_burgers_1d**: Rust implementation of the FDM on the 1D Burgers equation.
- **results**: Folder that stores the outputs of our FDM and PINN model in .csv format.
  - **results_1d_burgers.csv**
  - **results_fdm.csv**
- **burgers_eq_1_degree.py**: The main solver for the 1D Burgers equation.
- **burgers_eq_multi_variables.py**: The main solver for the 2D Burgers equation (to be modified).
- **data.py**: Class to create the training and validation datasets.
- **gradient_function.py**: Class to compute the gradient, Jacobian, and Hessian.
- **metrics.py**: Code that groups the different plots and metrics for analysis.
- **model.py**: Class to build and train the model.
- **test_1d_burgers_after_automated**: Test of the 1D Burgers code without automated hyperparameter tuning.



