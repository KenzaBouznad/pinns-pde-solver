import numpy as np
import tensorflow as tf
from data import DataGenerator
from gradient_function import Jacobian, Hessian, Gradient




class MyModel(tf.keras.Model):
    def __init__(self, n_inputs, n_outputs, hidden_units, activation):
        super(MyModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_units = hidden_units
        self.activation = activation
        self.build_layers() #this systematically triggers the build layers functions as soon as we define an element of class model

        
    def build_layers(self):
        self.model = tf.keras.models.Sequential()
        print('building layers yoooooooooooooooo')
        
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.n_inputs,))) #keras doesnt add the input layer explicitly so we add it using this keras module
        for unit in self.hidden_units:  # Build the hidden layers
            print(unit)
            self.model.add(tf.keras.layers.Dense(units=unit, activation=self.activation))
            print('layer added')

        self.model.add(tf.keras.layers.Dense(units=self.n_outputs))  # Output layer

        print(self.model.summary())


    def call(self, inputs, training=False):
        # Forward pass through the model
        return self.model(inputs, training=training)


class Training():
    def __init__(self, model: MyModel):
        self.model = model

    @tf.function
    def compute_domain_error(self, x_train, pde):

        y_pred = self.model.call(x_train, training=True)
        tf.shape(y_pred)
        jacobian_instance = Jacobian(self.model.call)
        hessian_instance = Hessian(self.model.call)
        domain_error = pde(x_train, y_pred, jacobian_instance, hessian_instance)

        domain_error = tf.cast(domain_error, tf.float64)
        return domain_error

    @tf.function
    def calculate_bc_loss(self, boundary_conditions, training_dataset):
        bc_loss = 0
        for dim in boundary_conditions.keys():
            for var, value, cond_type in boundary_conditions[dim]:
            
                x_ic_eval = tf.convert_to_tensor(value, dtype=tf.float64)
                x_ic_eval = tf.expand_dims(x_ic_eval, axis=0)
                
                x_ic = tf.convert_to_tensor([var], dtype=tf.float64)
                print(x_ic)
                x_ic = tf.expand_dims(x_ic, axis=0)
                print(x_ic)
                x_ic = tf.tile(x_ic, [1, self.model.n_inputs])
                print('the value of x after conversion is')
                print(x_ic)

                u_pred = self.model.call(x_ic, training=True)
                u_pred = tf.cast(u_pred, tf.float64)

                if cond_type == "neumann":
                    with tf.GradientTape() as condition_tape:
                        condition_tape.watch(x_ic)
                        du_pred = condition_tape.gradient(u_pred, x_ic)
                    bc_loss += tf.reduce_mean(tf.square(du_pred - x_ic_eval))
                else:
                    bc_loss += tf.reduce_mean(tf.square(u_pred - x_ic_eval))
        
        return bc_loss


    @tf.function
    def calculate_ic_loss(self, initial_conditions, x_initial_cond):
        ic_loss = 0
        for dim in initial_conditions.keys():
            for var, value, cond_type in initial_conditions[dim]:
                if callable(value): #this part (until else) is causing errors when running the code

                    x_ic_eval = value(x_initial_cond)  # Evaluate the function with x_ic

                    x_ic = x_initial_cond
                    x_ic = tf.concat([x_initial_cond, tf.zeros(shape=[x_initial_cond.shape[0], 1], dtype=tf.float64)], axis=1)
                    print("Evaluating callable IC function for x values:")
                    print(x_ic)
                    u_pred = self.model.call(x_ic, training=True) #we're adding the initial condition t=0

            

                    
                    print('The evaluated value of the initial condition is:')
                    print(x_ic_eval)

                else:
                    x_ic = tf.convert_to_tensor(value, dtype=tf.float64)
                    x_ic = tf.expand_dims(x_ic, axis=0)

                    x_ic_eval = value

                    u_pred = self.model.call(x_ic, training=True)

                
                u_pred = tf.cast(u_pred, tf.float64)

                ic_loss += tf.reduce_mean(tf.square(u_pred - x_ic_eval))
        
        return ic_loss

    def loss_reduction(self, training_dataset, validation_dataset, boundary_conditions, initial_conditions, epochs, learning_rate, pde):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        train_losses = []  # To track training losses
        val_losses = []    # To track validation losses
        
        total_mse = 0
        patience = 500  # Number of epochs to wait before early stopping
        best_val_loss = float('inf')  # Initialize with infinity
        epochs_no_improve = 0 

        for epoch in range(epochs):
            with tf.GradientTape() as model_tape:
                domain_error = self.compute_domain_error(training_dataset, pde)
                domain_mse = tf.reduce_mean(tf.square(domain_error))
                if epoch % 100 == 0:
                    print(f"the domain error for the epoch {epoch} is {domain_mse}")
                
                # Compute boundary/initial condition errors
                x_initial_cond = tf.Variable(training_dataset[:, 0:self.model.n_inputs-1])

                bc_loss = self.calculate_bc_loss(boundary_conditions, training_dataset)
                ic_loss = self.calculate_ic_loss(initial_conditions, x_initial_cond)

                total_mse = domain_mse + bc_loss + ic_loss

                # Compute gradients and apply optimizer step
                gradients = model_tape.gradient(total_mse, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
                # ===== Validation Step =====
            # Compute domain error for validation data
            domain_error_val = self.compute_domain_error(validation_dataset, pde)
            domain_mse_val = tf.reduce_mean(tf.square(domain_error_val))

            # Compute boundary and initial condition losses for validation data
            x_initial_cond_val = tf.Variable(validation_dataset[:, 0:self.model.n_inputs - 1])
            bc_loss_val = self.calculate_bc_loss(boundary_conditions, validation_dataset)
            ic_loss_val = self.calculate_ic_loss(initial_conditions, x_initial_cond_val)

            # Total validation loss
            total_val_mse = domain_mse_val + bc_loss_val + ic_loss_val

            # ===== Logging =====
            train_losses.append(total_mse.numpy())
            val_losses.append(total_val_mse.numpy())

            # Print losses every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}")
                print(f"  Training Loss: {total_mse.numpy()}")
                print(f"  Validation Loss: {total_val_mse.numpy()}")

                # ===== Early Stopping Logic =====
            if total_val_mse < best_val_loss:
                best_val_loss = total_val_mse  # Update the best validation loss
                epochs_no_improve = 0         # Reset the no improvement counter
            else:
                epochs_no_improve += 1  # Increment the no improvement counter

            # Check if early stopping should be triggered
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss.numpy()}")
                break

        return train_losses, val_losses
                    







    
    


        



