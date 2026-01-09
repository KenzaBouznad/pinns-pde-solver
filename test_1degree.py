import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
#from metrics import calculate_metrics
from matplotlib import pyplot as plt
from data import DataGenerator
from model import MyModel, Training
from gradient_function import Jacobian, Hessian



def problem_pde(x, y, jacobian_func, hessian_func): #this is defined according to the pde we want to solve
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32) 
  dy_x = jacobian_func.compute(x)


  dy_xx = hessian_func.compute_hessian(x)


  return dy_x - y

def main(batch_size, domain, n_inputs, n_outputs, hidden_units, condition_pair, activation, nb_layers, learning_rate, epochs) :

    '''in this test we're going to solve the burgers equation dy/dx = y with y(0) = 1 for x in [0,2]'''

    
    #first we define the dataset 

    dataset = DataGenerator(domain, batch_size, True) #we're defining a random dataset
    print(dataset)

    (training_dataset, testing_dataset) = DataGenerator.split_dataset(dataset.data)
    training_dataset, testing_dataset = np.array(training_dataset), np.array(testing_dataset)
    print(training_dataset, testing_dataset)

    #we define the model
    
    training_model = MyModel(n_inputs, n_outputs, hidden_units, nb_layers, activation)
    # Compile the model with a placeholder loss function
    training_model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    #we start training the model
    training_instance = Training(training_model)
    training_instance.loss_reduction(training_dataset, condition_pair, epochs, learning_rate, problem_pde)
  
    # evaluate the model on test data
    y_pred = training_model.model.predict(testing_dataset)
    print(y_pred)

    test_loss = training_model.model.evaluate(testing_dataset, y_pred)
    print(f'Test Loss: {test_loss}')

    y_true = np.exp(testing_dataset)
    print(y_true)

    print(y_pred-y_true)

    plt.plot(y_true)
    plt.plot(y_pred)
    plt.title('Evaluation')
    plt.legend(['Real', 'Predicted'])
    plt.show()
     





if __name__ == "__main__":
    # Define constants
    batch_size = 32
    domain = {'x': (0, 2)}
    n_inputs = len(domain.keys())
    n_outputs = 1  # in the case of a system of PDEs, we have n_outputs > 1
    hidden_units = [32, 32]
    condition_pair = [(0, 1, 'dirichlet')]  # this signifies y(0) = 1
    activation = 'tanh'
    nb_layers = 2
    learning_rate = 0.0005
    epochs = 5000

    # Call the main function with constants
    main(batch_size, domain, n_inputs, n_outputs, hidden_units, condition_pair, activation, nb_layers, learning_rate, epochs)

   
              





















