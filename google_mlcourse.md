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
  - Stochastic gradient descent> the loss function update happens for every randomly selected sample
  - Mini-batch gradient descent> the loss function gets updated based on the average taken over a set of samples
  - for multiple features, the gradient will be a vector of partial derivatives with respect to weights
  - -(del)f denotes the direction of greatest decrease of f
  - therefore gradient descent algorithm relies on -(del)f
  - gradients are calculated for all the weights (wi) and biases (bi)
   
Note:
- For linear regression problems, starting values of the weights and baises are not important
- Typically a regression problem yields convex loss vs. weight plot

Steps:
- start with an initial set of weights and biases
- for the given value of the feature set, compute the prediction function value (y')
- compute the loss function value based on the label and prediction function
- update the weights and the biases and perform the previous steps
- the model converges when the loss function changes very slowly 

