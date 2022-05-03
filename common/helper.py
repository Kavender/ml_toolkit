from typing import Iterable, Any
import json
import random
from numpy.random import uniform
from numpy import float64

def random_uniform(rng, shape=(), dtype=float64, minval=0.0, maxval=1.0):
  del rng
  return uniform(minval, maxval, size=shape).astype(dtype)


def uniform_seed(*args, **kwargs):
    return random_uniform(*args, **kwargs)


def load_json(path_to_file: str):
    "Helper function to load a JSON file"
    with open(path_to_file) as file:
        db = json.load(file)
    return db

# TODO: add ability to select random item with or without replacement
def stream(data: Iterable[Any], batch_size: int, shuffled: bool):
    # generator function to yield elements from input data
    if shuffled:
        random.shuffle(data)
    while True:
        # get a random list of unique elements chosen from input data
        d = random.sample(data, batch_size)
        # yield batch of random select set 
        yield d


def data_generator(batch_size, x, y, pad, shuffle=False, verbose=False):
    '''
      Input: 
        batch_size - integer describing the batch size
        x - list containing sentences where words are represented as integers
        y - list containing tags associated with the sentences
        shuffle - Shuffle the data order
        pad - an integer representing a pad character
        verbose - Print information during runtime
      Output:
        a tuple containing 2 elements:
        X - np.ndarray of dim (batch_size, max_len) of padded sentences
        Y - np.ndarray of dim (batch_size, max_len) of tags associated with the sentences in X
    '''
    
    # count the number of lines in data_lines
    num_lines = len(x)
    
    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]
    
    # shuffle the indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(lines_index)
    
    index = 0 # tracks current location in x, y
    while True:
        buffer_x = [0] * batch_size # Temporal array to store the raw x data for this batch
        buffer_y = [0] * batch_size # Temporal array to store the raw y data for this batch
                
  ### START CODE HERE (Replace instances of 'None' with your code) ###
        
        # Copy into the temporal buffers the sentences in x[index : index + batch_size] 
        # along with their corresponding labels y[index : index + batch_size]
        # Find maximum length of sentences in x[index : index + batch_size] for this batch. 
        # Reset the index if we reach the end of the data set, and shuffle the indexes if needed.
        max_len = 0
        for i in range(batch_size):
             # if the index is greater than or equal to the number of lines in x
            if index >= num_lines:
                # then reset the index to 0
                index = 0
                # re-shuffle the indexes if shuffle is set to True
                if shuffle:
                    rnd.shuffle(lines_index)
            
            # The current position is obtained using `lines_index[index]`
            # Store the x value at the current position into the buffer_x
            buffer_x[i] = x[lines_index[index]]
            
            # Store the y value at the current position into the buffer_y
            buffer_y[i] = y[lines_index[index]]
            
            lenx = len(buffer_x[i])    #length of current x[]
            if lenx > max_len:
                max_len = lenx                   #max_len tracks longest x[]
            
            # increment index by one
            index += 1


        # create X,Y, NumPy arrays of size (batch_size, max_len) 'full' of pad value
        X = np.full((batch_size, max_len), pad)
        Y = np.full((batch_size, max_len), pad)

        # copy values from lists to NumPy arrays. Use the buffered values
        for i in range(batch_size):
            # get the example (sentence as a tensor)
            # in `buffer_x` at the `i` index
            x_i = buffer_x[i]
            
            # similarly, get the example's labels
            # in `buffer_y` at the `i` index
            y_i = buffer_y[i]
            
            # Walk through each word in x_i
            for j in range(len(x_i)):
                # store the word in x_i at position j into X
                X[i, j] = x_i[j]
                
                # store the label in y_i at position j into Y
                Y[i, j] = y_i[j]

    ### END CODE HERE ###
        if verbose: print("index=", index)
        yield((X,Y))

# TODO: update docstring
def data_generator(X, y, batch_size, pad, shuffle, verbose=False):
    """
      Input: 
        X - list containing sentences where words are represented as integers
        y - list containing tags associated with the sentences
        batch_size - integer describing the batch size
        shuffle - Shuffle the data order
        pad - an integer representing a pad character
        verbose - Print information during runtime
      Output:
        a tuple containing 2 elements:
        X - np.ndarray of dim (batch_size, max_len) of padded sentences
        y - np.ndarray of dim (batch_size, max_len) of tags associated with the sentences in X
    """
    input_X = []
    input_y = []
    num_inputs = len(X)
    assert num_inputs == len(y) # two input sequence should be equal size

    # create an array with the indexes of data_lines that can be shuffled
    input_index = [*range(num_lines)]
    if shuffle:
        random.shuffle(input_index)
    
    idx = 0
    while True:
        if idx >= num_inputs:
            # if idx is greater than or equal to num_inputs, set idx accordingly 
            idx = 0
            if shuffle:
                random.shuffle(input_index)
        

