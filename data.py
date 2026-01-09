'''
here we're going to prepare the datasets used for training and testing. 
we want to create an array with the x, y, t values, split the array into testing and training datasets,
 
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataGenerator :
    def __init__(self, domain, nb_points, random) :
        
        self.domain = domain
        self.nb_points = nb_points
        self.random = random
        self.data = self.Dataset()
    
    def Dataset(self) :

        '''
            If you are solving a problem where you need data across the entire domain in a systematic,
              regular manner (such as for a physical simulation or PDE), you would prefer using 
              meshgrid to generate your data points. This ensures that the data covers the domain 
              in a structured way, allowing you to accurately evaluate or compute quantities at each 
              grid point.'''
        
        print(self.domain.keys())
        
        variables = list(self.domain.keys())
        nb_variables = len(variables)
        print(nb_variables)

        data_var=pd.DataFrame()

        
        if self.random == True:
            for var in variables :
                (min, max) = self.domain[var]
                print(var, min, max)
                data_var[f'{var}'] = np.random.uniform(min, max, self.nb_points)
                data = data_var
                print(data)
                #data = pd.concat([data, data_var], axis=1) #this might need to be transposed
            
        else:
            grids = np.meshgrid(*[np.linspace(min_val, max_val, int(self.nb_points**(1/nb_variables))) 
                                  for min_val, max_val in [self.domain[var] for var in variables]], indexing="ij")
            data = pd.DataFrame(np.vstack([grid.ravel() for grid in grids]).T, columns=variables)
        
        return data
    
    
    def split_dataset(data):
            """
            Splits the dataset into training and testing subsets.
            """
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Expected a pandas DataFrame as input for splitting.")
            train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)
            return train_data, test_data
    
    def __repr__(self):
        return f"DataGenerator(domain={self.domain}, nb_points={self.nb_points}, random={self.random})"
    
    
    

        

'''grid.ravel():

This takes each grid (which is an array representing one dimension of the meshgrid) and 
flattens it into a 1D array. For example, if grid is a 2D array, ravel() will return a 1D 
array with the same elements as the original 2D array but in a single row.
[grid.ravel() for grid in grids]:

This is a list comprehension that iterates over each grid in grids (where grids is a list of 
arrays representing the dimensions of the meshgrid). For each grid, it applies ravel() to flatten it.
The result is a list of flattened 1D arrays (each representing a variable's mesh).
np.vstack([...]):

np.vstack() takes the list of 1D arrays generated in the previous step and stacks them vertically 
(i.e., one on top of the other) to create a 2D array.
Each row in the resulting 2D array corresponds to a point in the multi-dimensional grid, 
with each column corresponding to one of the original variables.
.T (transpose):

.T transposes the 2D array. Transposing means converting the rows into columns and vice versa.
In this context, transposing the array ensures that the columns correspond to the individual 
variables, while each row corresponds to a point in the multi-dimensional space.'''


