import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
#from metrics import calculate_metrics
from matplotlib import pyplot as plt
from data import DataGenerator
from model import MyModel, Training
from gradient_function import Jacobian, Hessian, Gradient
import optuna



def problem_pde(x, y, jacobian_func, hessian_func): #this is defined according to the pde we want to solve
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32) 

    
    dy_x = jacobian_func.compute_jacobian(x)

    dy_xx = hessian_func.compute_hessian(x)

    print("dy_x:", dy_x)
    print("dy_x shape:", dy_x.shape if dy_x is not None else None)

    print("dy_xx:", dy_xx)
    print("dy_xx shape:", dy_xx.shape if dy_xx is not None else None)


    return dy_x[:,:,0] + y * dy_x[:,:,1] - (0.01/np.pi)*dy_xx[:, :, 0, 1]

def main(batch_size, domain, n_inputs, n_outputs, hidden_units, boundary_conditions, initial_conditions, activation, learning_rate, epochs) :

    '''in this test we're going to solve the 1D burgers equation using the Optuna optimized hyperparameters'''

    
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

    domain = {'x': (0, 1), 't': (0, 1)}
    test_dataset = DataGenerator(domain, batch_size, False) #we're defining a random dataset
    # Extract input features from the test dataset
    x_test = test_dataset.Dataset() 


    test_results = training_model.model.predict(x_test)
    test_results = pd.DataFrame(test_results)

    test_results = test_results.rename(columns={0: 'u'})
    x_test = pd.DataFrame(x_test, columns=["x", "t"]) 

    test_results = pd.concat([x_test, test_results], axis=1)

    print(test_results)

    test_results.to_csv('./results/results_1d_burgers.csv', index=False)
    print("Test results saved to 'test_results_1d_burgers.csv'.")




if __name__ == "__main__":


    # Extract best hyperparameters
    n_layers = 5
    n_units = 32    
    activation = 'relu'
    learning_rate = 0.0032826089223827386
    
    # Run main function with best parameters
    hidden_units = [n_units for _ in range(n_layers)]

    # Define constants
    batch_size = 1000
    domain = {'x': (-1, 1), 't': (0, 1)} #the domain for t doesnt include 0
    n_inputs = len(domain.keys())
    n_outputs = 1  # in the case of a system of PDEs, we have n_outputs > 1
 
    boundary_conditions = {
        'x': [(-1, 0, 'dirichlet'), (1, 0, 'dirichlet')],}

    initial_conditions = {
    'u': [(0, lambda x: -tf.sin(np.pi * x), 'other')]
    }

    epochs = 15000

    # Call the main function with constants
    main(batch_size, domain, n_inputs, n_outputs, hidden_units, boundary_conditions, initial_conditions, activation, learning_rate, epochs)

