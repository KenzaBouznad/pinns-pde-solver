import tensorflow as tf
import numpy as np

class Gradient:
    "Computes the gradient of a function and its jacobian, hessian and other gradient functions."
    def __init__(self, function):
        # Initializing the function
        self.function = function
    
    def gradient(self, variable):
        variable = tf.convert_to_tensor(variable)  # Ensure variable is a tensor
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(variable)
            outputs = self.function(variable)
            gradients = tape.gradient(outputs, variable)
        return tape, outputs, gradients

class Jacobian(Gradient):
    def __init__(self, func):
        """
        Initializes the Jacobian class.
        
        Args:
            func: A callable function, typically the neural network's forward pass.
        """
        super().__init__(func)  # Call the parent class's constructor

        self.func = func

    def compute_jacobian(self, x):
        """
        Computes the Jacobian of the function with respect to its input.
        
        Args:
            x: Input tensor.
            
        Returns:
            Jacobian tensor of shape (batch_size, output_dim, input_dim).
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.func(x)
        return tape.jacobian(y, x)

    
class Hessian(Gradient):
    def __init__(self, func):
        super().__init__(func)  # Call the parent class's constructor

        """
        Initializes the Hessian class.
        
        Args:
            func: A callable function, typically the neural network's forward pass.
        """
        self.func = func

    def compute_hessian(self, x):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(x)
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(x)
                y = self.func(x)
                gradients = inner_tape.gradient(y, x)
            hessian = outer_tape.jacobian(gradients, x)
        return hessian
