1. ML concepts

[A] Framing ML
- Labels :
- Features :
- Examples :
 - Labeled example
 - Unlabeled example
-Models:
 - Training a model
   - prediction function (model) gets trained using the identified labels
   - prediction function is a mapping between the features and the labels
   - through training, the weights and baises in the prediction function get optimized to minimize loss function
   - the minimization happens through what is usually known as ML algorithms (?)
 - Inference
-Regression vs Classification :

[B] Descending into ML
- Linear Regression :
  - the y-intercept is the bias 
  - the slope is the weight 
  - x represents a feature
  - y represents a label
  - there can be multiple features such as x1, x2, x3,...
 
- Loss :
  - quantifing error in terms of L2 loss parameter where it is squared difference between target and prediction
  - negative gradient of loss function with respect to the features will give the direction for optimum loss reduction
  - the step size or which is also called learning rate is an important parameter
    - very small learning rate is computionally costly
    - a large learning rate may overshoot an optimum state
  - initialization plays an important role 

- Gradient Descent :
  - Stochastic gradient descent (SGD)> the loss function update happens for every randomly selected sample
    - the batch size is 1
    - it is noisy but works given enough iteration
  - Mini-batch gradient descent> the loss function gets updated based on the average taken over a set of samples
    - the batch size is between 10 to 1000
    - mini-batch SGD reduces the amount of noise in SGD but is still more efficient that full-batch
    - full-batch usually have redundant data
  - for multiple features, the gradient will be a vector of partial derivatives with respect to weights
  - -(del)f denotes the direction of greatest decrease of f
  - therefore gradient descent algorithm relies on -(del)f
  - gradients are calculated for all the weights (wi) and biases (bi)
  - gradient descent algorithms multiply the gradient by a scalar known as the learning rate (step size)
  - the next point then is at a distance of (the gradient times the learning rate)
  - learning rate is an example of a hyperparameter (usually learning rate is between 0 and 1)
  - hyperparameters are the knobs that programmers tweak in ML algorithms
  - batch size is another hyperparameter
   
Note:
- For linear regression problems, starting values of the weights and baises are not important
- Typically a regression problem yields convex loss vs. weight plot
- the ideal learning rate in 1D is (1/(f(x)''))
- for higher dimensional problems, the ideal learning rate is the inverse of the Hessian
- Learn about Hessian

Steps:
- start with an initial set of weights and biases
- for the given value of the feature set, compute the prediction function value (y')
- compute the loss function value based on the label and prediction function
- update the weights and the biases and perform the previous steps
- the model converges when the loss function changes very slowly 

Tuning Hyperparameters:
 - For basic problems, we have batch size, epoch number, and learning rate as the tuning hyperparameters
 - epoch number is different from iterations, wherein, at each epoch we can have one or more iterations based on whether the    mini-batch size overlaps with the full batch size
     - if mini-batch size is equal to the full batch size, we have one iteration per epoch
     - if mini-batch size is smaller than full batch size, we have (full-batch size/mini-batch size) iterations
 - tuning of hyperparameters is specific to each ML problem, and batch-size typically depends on the number of samples          (population size/sample size)
 - if the loss curve doesn't appear to converge, increase the number of epochs
 - if the loss curve has ups and downs,change or reduce the learning rate
 - usually mini-batch size (also known in general as batch size) is in powers of 2. Nevertheless, try odd numbers as well.
 - beware of false convergence (just made-up the word). Basically, the loss curve seems to converge but actually settles on    a sub-optimal value 

Tasks:
- look for gradient descent algorithm in Python
- try to customize it for a specific use

Crazy Ideas:
 - experiment with complex neuron values 
 
Numpy:
  - It is a python library for creating and manipulating vectors and matrices
  - to import Numpy module > import numpy as np
  - np.array is used to create matrices:
    - 1D matrix containing five elements > matrix_1D_example = np.array([1.2, 3, 4, 5.6, 8.3])
    - 2D matrix (3x2) containing six elements > matrix_2D = np.array([[6, 5], [11, 7], [4, 8]])
    - notice the extra pair of square brackets in the 2D matrix definition
    - to create a matrix with all zeros > np.zeros
    - to create a matric with all ones > np.ones
    - to create a vector of integers in sequence > np.arange(6, 12)    # notice that 12 doesn't appear in the sequence
  - np.random.randint is used to generate random integers between a low and high value:
    - random_integers_bandlimited = np.random.randint(low=50, high=101, size(6))
    - notice that the highest generated integer with be one less than the high argument
  - np.random.random is used to generate random floating-point values between 0 and 1:
    - random_floatingpoint = np.random.random([8])
  - Numpy uses a technique called broadcasting technique wherein matrix operations become easier
    - smaller dimension matrices are automatically scaled up to match the matrix with higher dimension
    - this takes care of dimensional compatibility
    
  - importing Numpy and pandas modules:
    - import numpy as np
    - import pandas as pd
    
  - DataFrames are the defacto data structure in the pandas API
    - like an in-memory spreadsheet, a DataFrame stores data in cells
    - A DataFrame has named columns and numbered rows
    
  - Creating a simple DataFrame:
    - my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])   # np.array is called to create a 5x2 Nympy array
    - column_names = ['temperature', 'activity']   # a Python list is created to hold the names of the columns
    - my_dataframe = pd.DataFrame(data = my_data, columns = column_names)   # class pd.DataFrame is instantiated 
    - my_dataframe['adjusted'] = my_dataframe['activity'] + 2   # adding a new column to the existing pandas DataFrame
  
  - Isolating specific rows, columns in a DataFrame:
    - print(my_dataframe.head(3), '\n')    # prints rows #0, #1, and #2
    - print(my_dataframe.iloc[[2]], '\n')   # prints row #2
    - print(my_dataframe[1:4], '\n')   # prints rows #1 to #3
    - print(my_dataframe.loc[row_number,'column_name'])   # identifies a specific element
    - print(my_dataframe['column_name'])   # prints a specific column
    
    
Note:
- Low loss for a given dataset doesn't necessarily mean that the model is doing good
- Adaptation to a new dataset is important for model performance
- Overfitting a model for a given dataset is not good for generalization
- An overfit model gets a low loss during training but does a poor job predicting new data
- Overfitting is caused by making model more complex than necessary
- The mantra is to fit the data well but also as simple as possible
- Machine learning's goal is to predict well on new data drawn from a (hidden) true probability distribution. 
- Ockham's razor in machine learning: The less complex an ML model, the more likely that a good empirical result is not just due to the pecularities of the sample.
- Ideally we would like to train our model on a data sample and know that it is going to do well on new draws of data samples from a hidden distribution
- A dataset can be divided into two: training set (a subset to train a model), and a test set (a subset to test the model)
- We are not concerned with the process which generates data, rather, we work with data samples generated from a process
- Test set methodology: We pull one draw of data from a distribution and we train on that. That is our training set. We take another draw of data from that distribution which we call as a test set.
    - Basic assumptions:
     - examples are drawn independently and identically (i.i.d) at random from a given distribution
     - The distribution remains stationary
     - The data is taken from the same distribution: including traning, validation, and tests
          
Test set and Training set methodology:
 - Test set must meet the following two conditions:
   - is large enough to yield statistically meaningful results
   - is representative od the dataset as a whole. That means, the training and test sets should have similar characteristics
   - example: (changing the learning rate)
     - learning rate = 3; epochs = 300; traning data percentage = 50%; Noise = 80; Batch size = 1; 
       - Test loss = 0.278; Training loss = 0.208; Difference  = |test loss - training loss| = 0.07
     - learning rate = 1; epochs = 302; samee data parameters as before;
       - Test loss = 0.262; Training loss = 0.208; Difference = 0.054
     - learning rate = 0.3; epochs = 305; same data parameters as before;
       - Test loss = 0.299; Training loss = 0.213; Difference = 0.086
     - learning rate = 0.1; epochs = 300; same data parameters as before;
       - Test loss = 0.227; Training loss = 0.181; Difference = 0.046
     - learning rate = 0.03; epochs = 296; same data parameters as before;
       - Test loss = 0.220; Training loss = 0.176; Difference = 0.044
     - learning rate = 0.01; epochs = 301; same data parameters as before;
       - Test loss = 0.211; Training loss = 0.173; Difference = 0.038
     
   - example: (changing learning rate for a batch size of 10)
     - learning rate = 3; epochs = 308; batch size = 10; Training data percentage = 50%; Noise = 80
       - Test loss = 0.263; Training loss = 0.204; Difference = 0.059
     - learning rate = 1; epochs = 300; same data parameters as before;
       - Test loss = 0.252; Training loss = 0.188; Difference = 0.064
     - learning rate = 0.3; epochs = 300; same data parameters as before:
       - Test loss = 0.222; Training loss = 0.176; Difference = 0.046
     - learning rate = 0.1; epochs = 302; same data parameters as before:
       - Test loss = 0.212; Training loss = 0.173; Difference = 0.039
     - learning rate = 0.03; epochs = 305; same data parameters as before:
       - Test loss = 0.208; Training loss = 0.172; Difference = 0.036
     - learning rate = 0.01; epochs = 306; same data parameters as before:
       - Test loss = 0.207; Training loss = 0.172; Difference = 0.035
   
   - example: (changing learning rate, for a batch size of 10, and training data percentage of 10%)
     - learning rate = 3; epochs = 304; batch size = 10; Training data percentage = 10%; Noise = 80
       - Test loss = 0.357; Training loss = 0.121; Difference = 0.236
     - learning rate = 1; epochs = 307; same data parameters as before;
       - Test loss = 0.353; Training loss = 0.121; Difference = 0.232
     - learning rate = 0.3; epochs = 300; same data parameters as before;
       - Test loss = 0.342; Training loss = 0.125; Difference = 0.217
     - learning rate = 0.1; epochs = 301; same data parameters as before;
       - Test loss = 0.324; Training loss = 0.134; Difference = 0.19
     - learning rate = 0.03; epochs = 302; same data parameters as before;
       - Test loss = 0.286; Training loss = 0.150; Difference = 0.136
     - learning rate = 0.01; epochs = 301; ame data parameters as before;
       - Test loss = 0.260; Training loss = 0.160; Difference = 0.100
   
   - outcome from examples:
     - reducing learning rate reduces the difference between the Training loss and Test loss
     - increasing the batch size slighlty reduces the difference
     - reducing Training data percentage to 10% increases both the losses and difference
     
   - Note:
     - The more often we train our model on the given test set, the more the rist of overfitting the model to the one test          set
