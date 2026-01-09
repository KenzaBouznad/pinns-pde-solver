# Time complexity and compare it to someone elses work from literature
# Numbering pages
# local optima and global optima zith adam optimizer | adam works best with stochastic 
# what does convection and diffusion mean and how does that affect calculation and time complexity
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from data import DataGenerator
from model import MyModel, Training
from gradient_function import Jacobian, Hessian, Gradient
import optuna
from keras.callbacks import EarlyStopping



def problem_pde(x, y, jacobian_func, hessian_func):
    """
    Defines the PDE based on inputs and outputs, and calculates derivatives using Jacobians and Hessians.
    
    Args:
        x: Input tensor (batch_size, 3) with x, y, t stacked together.
        y: Output tensor (batch_size, 2) representing u and v.
        jacobian_func: A function to compute Jacobians.
        hessian_func: A function to compute Hessians.
    
    Returns:
        Tensor of shape (batch_size, 2) containing the residuals of the PDE equations.
    """

    inputs = tf.cast(x, tf.float32)  # Inputs assumed to be x, y, t stacked together
    print("Inputs shape:", inputs.shape)
    outputs = tf.cast(y, tf.float32)  # Outputs assumed to represent [u, v]
    print("Outputs shape:", outputs.shape)

    # Pass inputs directly through jacobian and hessian
    d_outputs_combined = jacobian_func.compute_jacobian(inputs)  # Shape: (batch_size, 2, 3)
    print("d_outputs_combined shape:", d_outputs_combined.shape)
    d2_outputs_combined = hessian_func.compute_hessian(inputs)  # Shape: (batch_size, 2, 3)
    print("d2_outputs_combined shape:", d2_outputs_combined.shape)

    # Extract components for u and v from outputs
    u = outputs[:, 0:1]  # Shape: (batch_size, 1)
    v = outputs[:, 1:2]  # Shape: (batch_size, 1)

    # Gradients of u and v with respect to x, y, t
    du_x = d_outputs_combined[:, 0, 0:1]  # ∂u/∂x
    du_y = d_outputs_combined[:, 0, 1:2]  # ∂u/∂y
    du_t = d_outputs_combined[:, 0, 2:3]  # ∂u/∂t

    dv_x = d_outputs_combined[:, 0, 0:1]  # ∂v/∂x
    dv_y = d_outputs_combined[:, 0, 1:2]  # ∂v/∂y
    dv_t = d_outputs_combined[:, 0, 2:3]  # ∂v/∂t

    # Second derivatives of u and v with respect to x and y
    d2u_xx = d2_outputs_combined[:, 0, 0:1]  # ∂²u/∂x²
    d2u_yy = d2_outputs_combined[:, 0, 1:2]  # ∂²u/∂y²

    d2v_xx = d2_outputs_combined[:, 1, 0:1]  # ∂²v/∂x²
    d2v_yy = d2_outputs_combined[:, 1, 1:2]  # ∂²v/∂y²

    # Define the equations for Burgers' PDE
    eq1 = du_t + u * du_x + v * du_y - 0.01 / np.pi * (d2u_xx + d2u_yy)  # Residual for u
    eq2 = dv_t + u * dv_x + v * dv_y - 0.01 / np.pi * (d2v_xx + d2v_yy)  # Residual for v

    # Return concatenated residuals
    return tf.concat([eq1, eq2], axis=-1)  # Tensor shape: (batch_size, 2)

def main(batch_size, domain, n_inputs, n_outputs, hidden_units, boundary_conditions, initial_conditions, activation, learning_rate, epochs) :

    '''in this test we're going to solve the 2D burgers equation with sinusoidal initial conditions'''

    
#first we define the dataset 

    dataset = DataGenerator(domain, batch_size, True) #we're defining a random dataset
    print(dataset)

    (training_dataset, validation_dataset) = DataGenerator.split_dataset(dataset.data)
    training_dataset, validation_dataset = np.array(training_dataset), np.array(validation_dataset)
    print(training_dataset, validation_dataset)

    #we define the model
    
    training_model = MyModel(n_inputs, n_outputs, hidden_units, activation)

    # Compile the model with a placeholder loss function
    training_model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    #we start training the model
    training_instance = Training(training_model)

    #we fix the boundary and initial conditions
    
    train_loss, val_loss =training_instance.loss_reduction(training_dataset, validation_dataset, boundary_conditions, initial_conditions, epochs, learning_rate, problem_pde)
    
    # Plot training and validation losses (learning curves)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.show()

    # Evaluate the model on validation data
    validation_loss = val_loss[-1]  # Use the last recorded validation loss
    print(f"Final Validation Loss: {validation_loss}")

    #testing with a unifrorm mesh grid

    test_dataset = DataGenerator(domain, batch_size, False) #we're defining a random dataset
    # Extract input features from the test dataset
    x_test = test_dataset.Dataset() 


    test_results = training_model.model.predict(x_test)
    test_results = pd.DataFrame(test_results)

    test_results = test_results.rename(columns={0: 'u'})
    x_test = pd.DataFrame(x_test, columns=["x", "t"]) 

    test_results = pd.concat([x_test, test_results], axis=1)

    print(test_results)

    test_results.to_csv('./results/results_2d_burgers.csv', index=False)
    print("Test results saved to 'test_results_2d_burgers.csv'.")
     
def objective(trial):
    # Define hyperparameter search space
    n_layers = trial.suggest_int('n_layers', 1, 5)  # Number of hidden layers
    n_units = trial.suggest_int('n_units', 16, 128, step=16)  # Number of units per layer
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    # Create hidden_units list based on n_layers and n_units
    hidden_units = [n_units for _ in range(n_layers)]

    # Generate training and validation datasets
    domain = {'x': (-1, 1), 't': (0, 1)}
    batch_size = 1000
    dataset = DataGenerator(domain, batch_size, True)
    training_dataset, validation_dataset = DataGenerator.split_dataset(dataset.data)
    training_dataset, validation_dataset = np.array(training_dataset), np.array(validation_dataset)

    # Define model
    model = MyModel(
        n_inputs=len(domain.keys()),
        n_outputs=1,  # Assuming single-output PDE solution
        hidden_units=hidden_units,
        activation=activation
    )

    # Compile the model
    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    # Define EarlyStopping callback to monitor validation loss
    early_stopping = EarlyStopping(
        monitor='val_loss',       # Monitor validation loss
        patience=50,              # Stop after 50 epochs of no improvement
        restore_best_weights=True # Restore the best weights after stopping
    )


    # Train the model
    training_instance = Training(model)
    train_losses, val_losses = training_instance.loss_reduction(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        boundary_conditions={'x': [(-1, 0, 'dirichlet'), (1, 0, 'dirichlet')]},
        initial_conditions={'u': [(0, lambda x: -tf.sin(np.pi * x), 'other')]},
        epochs=200,  # Use fewer epochs for faster trial evaluation
        learning_rate=learning_rate,
        pde=problem_pde
    )

    # Return the final validation loss (objective to minimize)
    return val_losses[-1]  # Use the last validation loss as the metric


def optimize_hyperparameters(n_trials=50):
    # Create a study to minimize the objective function
    study = optuna.create_study(direction='minimize')
    
    # Optimize the study
    study.optimize(objective, n_trials=n_trials)

    # Print the best trial and hyperparameters
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best parameters: {study.best_trial.params}")
    print(f"Best validation loss: {study.best_trial.value}")

    return study.best_trial.params


if __name__ == "__main__":

    # Run hyperparameter optimization
    best_params = optimize_hyperparameters(n_trials=50)  # Adjust n_trials as needed

    # Extract best hyperparameters
    n_layers = best_params['n_layers']
    n_units = best_params['n_units']
    activation = best_params['activation']
    learning_rate = best_params['learning_rate']
    
    # Run main function with best parameters
    hidden_units = [best_params['n_units'] for _ in range(best_params['n_layers'])]

    # Define constants
    batch_size = 1000
    domain = {'x': (-1, 1), 'y': (-1, 1), 't': (0, 1)} #the domain for t doesnt include 0
    n_inputs = len(domain.keys())
    n_outputs = 2  # in the case of a system of PDEs, we have n_outputs > 1


    boundary_conditions = {
        'x': [(-1, 0, 'dirichlet'), (1, 0, 'dirichlet')],
        'y': [(-1, 0, 'dirichlet'), (1, 0, 'dirichlet')],
    }
    initial_conditions = {
    'u': [(0, lambda x: -tf.sin(np.pi * x), 'other')],
    }

    epochs = 3000

    # Call the main function with constants
    main(batch_size, domain, n_inputs, n_outputs, hidden_units, boundary_conditions, initial_conditions, activation, learning_rate, epochs)

   
              
