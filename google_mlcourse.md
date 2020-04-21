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
    
