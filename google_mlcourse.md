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
    
  - shuffling a dataset:
    - train_df = train_df.reindex(np.random.permutation(train_df.index))
    
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
    - new_df['Column_name'] /= scaling_factor   # scaling operation
    
    
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
     
   - Validation set:
     - The more often we train our model on the given test set, the more the risk of overfitting the model to the one test          set
     - The effect of overfitting on test set can be minimized by first testing the model on a third set, which is the              validation set
     - We then try to minimize the difference in performance of the model on the validation set and test set
     - A good performance on validation set and not on a test set may indicate overfitting
     - The strategy is to pick the model that does best on the validation set, and then test on test set
     - Test sets and validation sets wear out with repeated use. That is, the more you use the same data to make decisions          about hyperparameter settings, the less confidence will be on the model to do well on new test set
     - Validation set is carved out from a given training set (is it always?)
     - One must randomize the training set to eliminate any bias that may result on differences in training and validation          set
     - to shuffte a data set, one can do:
       - shuffled_df = my_df.reindex(np.random.permutation(my_df.index))
     - shuffling reduces the difference between the traning loss and validation loss
     - The test loss then will be similar to the training loss and validation loss

  - Representation
    - take data from heterogeneous data sources and create feature vectors from it
    - raw data doesn't come to us as feature vectors
    - The process of extracting features from raw data is called feature engineering
    - If we see a string value, we can often translate that into a feature vector by using a one-hot encoding
    - Categorical features have a discrete set of posible values. Collection of these possible values is vocabulary
    - One-hot encoding comprises of a binary vector whose length is equal to the number of elements in vocabulary, and for a       specific possible value, the corresponding binary vector element is set to 1 and all others are set to zero.
    - One-hot encoding extends to numeric data that may not be appropriate to get directly multiplied with by weights
    - A good feature should occur with a non-zero value at least a handful of times in our data set
    - If a feature with a non-zero value occurs only extremely rarely or even once, its probably not a good feature to use,       and should be filtered out in a pre-processing step
    - Histograms, and statistical features like maximum and minimum values, mean, median, standard deviation can inform           about the features of the data set
    - Binning of data in a dataset adds in more sub-categories which may help in indentifying more intrinsic characteristics       about the data
    - Binning data by quantiles is a good approach, which enables equal distribution of data in each bin
    - scaling feature values means converting their natural range to a standard range of 0 to 1, or, -1 to +1.
    - Scaling helps gradiant decent to converge more quickly, avoids ill-conditioning and, helps the model learn appropriate       weight for each feature
   
  - Feature crosses
    - A feature cross is a synthetic or derived feature formed by multiplying (crossing) two or more features. Cross               combinations of features can provide predictive abilities beyond the capabilities of the individual features
    - In other words, feature crossing (for example, [AxB]) helps us to learn nonlinearity in a linear model using a               synthetic feature
    - A feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input       features together. The term cross comes from cross product.
    - y = b + w1x1 + w2x2 + w3(x1*x2) = b + w1x1 + w2x2 + w3x3
    - When A and B represent boolean features, such as bins, the resulting crosses can be extremely sparse
    - Feature crosses help us to incorporate nonlinear learning into a linear learner
    - linear models scale well to massive data, but without feature crosses, the expressibility of these models would be           limited
    - using feature crosses + massive data (and deep neural networks) is one efficient way for learning highly complex             models 
    - Supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-         scale data sets
    - In practice, ML models seldom cross continuous features. However, ML models do frequently cross one-hot feature             vectors
    - Crossing a feature vector of side nx1 and another feature vector of size mx1, creates a synthetic feature vector of         size (n*m)x1
    - The model output surface for a linear model with synthetic features resembles a nonlinear shape
    
    - Feature column creation:
      - tf.feature_column method represents a single feature, or a single feature cross, or a single synthetic feature in a         desired way
      - for representing a single feature as floating-point values, call tf.feature_column.numeric_column
      - for representing a single feature as a series of buckets or bins, call tf.feature_column.bucketized_column
      - for example, to create a numeric feature column to represent latitude:
         - latitude = tf.feature_column.numeric_column("latitude")
         - feature_columns = []  # an empty list is created to hold all feature columns
         - feature_columns.append(latitude)
   
    - Regularization: Simplicity
      - If we use a model with too many crosses, we give the model opportunity to fit to the noise in the training data,             often at the cost of making the model perform badly on test data
      - Regularization - not trusting the examples in training data too much
      - Regularization is done to avoid overfitting
        - Early stopping: ending training before the model fully reaches convergence
        - Trying to penalize model complexity - also known as structural risk minimiation
          - Model complexity can be done by prefering smaller weights
            - done using L2 regularization (a.k.a ridge regularization)
              - In this penalization strategy, we penalize the sum of the squared values of the weights
              - A loss function with L2 regularization:
                - TrainingLoss + (lambda)*(w1^2 + w2^2 + w3^2 + .... + wn^2)
                - The second term (regularization term) in the loss function doesn't depend on the data
                - (lambda) dictates the balance between getting the examples right and keeping the model simple
                - Lamba is known as regularization rate
                - Lamba is used to tune the overall impact of the regularization term
                - For a model complexity as a function of weights, a feature weight with a high abosolute value is more                       complex than a feature weight with a low absolute value
                - In L2 regularization, weights close to zero have little effect on model complexity, while outlier weights                   can have a huge impact
                - regularization is minimally required when traning data and test data are similar
                - when there is not much of a training data, or when the training data and test data are kind of different,                   then regularization is required a lot
                - L2 regularization has the following effect on a model:
                  - encourages weight values towards 0 (but not exactly zero)
                  - encourages the mean of the weights toward 0, with a Normal distribution
                - increasing Lambda value strengthens the regularization effect
                - The ideal value of Lambda produces a model that generalizes well to new, previously unseen data. This                       ideal value of Lambda is data-dependent, and so, will require tuning
                - including and/or increasing regularization rate reduces test loss, while training loss increases. This is                   expected, as we have added an extra term to the loss function to penalize complexity. Ultimately, all that                   matters is test loss, as that's the true measure of the model's ability to make good predictions on new                     data
                - including and/or increasing regularization rate reduces the difference between training loss and test loss
                
      - Learning rate and L2 regularization:
        - Strong L2 regularization values tend to drive feature weights closer to zero
        - Lower learning rates (with early stopping) aften produce the same effect 
        - Simultaneously tweaking learning rate and Lamba may have profound effects
                
    
    - Note: A generalization curve shows the loss for both the traning set and validation set against the number of traning       iterations
    
    - Logistic Regression:
      - It is a prediction method that gives us well calibrated probabilities
         - Logistic regression is an extremely efficient mechanism for calculating probabilities
         - we can use the returned probability either "as is", or convert it to a binary category
         - in many cases, we will map the logistic regression output into the solution to a binary classification problem
      - We take our linear model and stick it in to a sigmoid function
      - sigmoid gives us a bounded value between zero and one
      - we train the model using a LogLoss function, not a squared loss function
        - the loss function for linear regression is squared loss
        - the loss function for logistic regression is LogLoss, defined as:
          - LogLoss = summation(-ylog(y') - (1-y)log(1-y')
            - where, y is the label in a labeled example. Since, this is logistic regression, every value of y is between 0               and 1
            - y' is the predicted value (somewhere between 0 and 1)
      - LogLoss resembles Shanon's entropy measure from information theory
      - The asymptotes of the sigmoid function are important in terms of learning
      - These asymptotes make it necessary to incorporate regularization in to learning, otherwise, on a given dataset the           model will try to fit our data ever more closely. (trying to drive the losses to near zero)
      - L2 regularization can be extremely helpful here to make sure that the weights don't go crazy out of bounds
      - Linear Logistic regression is very fast, extremely efficient to train, and efficient to make predictions
        - it scales well to massive data
        - can be used for very low latency predictions
        
      - A sigmoid function is given as:
        - y = 1/(1 + e^-(z)) 
        - y lies between 0 and 1, and approaches 0 and 1 asymtotically
        - If z represents the output of the linear layer of a model trained with logistic regression, then sigmoid(z) will             yield a value (a probability) between 0 and 1. y in that case is the output of the logistic regression model for             the given example
        - z = b + w1x1 + w2x2 + w3x3 +....+wnxn
        - z is also known as the log-odds because the inverse of the sigmoid states that z can be defined as the log of the           probability of the "1" label (something happens) divided by the probability of the "0" label (something doesn't             occur)
          - z = log(y/(1-y))
          
    - Classification:
      - we can use logistic regression as a foundation for classification by taking the probability outputs and applying a           fixed classification threshold (also called the decision threshold) to them
      - tuning a threshold for logistic regression is different from tuning hyperparameters such as learning rate. Part of           selecting a threshold is assessing how much the cost of a mistake will be
      - to quantify classification performance, one classical way is to use accuracy
        - accuracy is the fraction of predictions we got right
        - accuracy breaks down when there is class imbalance in a problem, where there is a significant disparity between             the number of positive and negative labels
      - for class-imbalance problems, it is useful to separate out different kinds of errors as:
        - True positives (TP); False positives (FP); False negatives (FN); True negatives (TN)
        - consider that "spam" is a positive class, and "not spam" is a negative class
        - TP is an outcome where the model correctly predicts the positive class. TN is an outcome where the model correctly           predicts the negative class
        - FP is an outcome where the model incorrectly predicts the positive class. FN is an outcome where the model                   incorrectly predicts the negative class
        - accuracy = (number of correct predictions)/(total number of predictions) = (TP+TN)/(TP+TN+FP+FN);
        - we can combine these ideas into a couple of different metrics:
          - Precision: (TP)/(TP + FP)
            - In general, raising the classification threshold reduces FP, thus raising precision
          - Recall: (TP)/(TP + FN)
            - raising classification threshold will cause the number of TP to decrease or stay the same, and will cause the               number of FN to increase or stay the same. Thus recall will either stay constant or decrease
          - In general, a model that outperforms another model on both precision and recall is likely the better model
          - these two metrics are often at tension and doing well at both of them is important. 
          - improving precision typically reduces recall and vice-versa.
          - it is important to know both precision and recall before assessing the quality of a given model
          - raising the classification threshold typically increases precision: however, precision is not guaranteed to                 increase monotonically as we raise the threshold
          - precision and recall are both well-defined when there is one specific classification threshold that we have                 chosen
          - we have a metric that looks at the performance of the model across a range of classification threshold. It is               known as receiver operating characteristics (ROC) curve
            - ROC curve is a graph showing the performance of a classification model at all classification thresholds
            - an ROC curve plots two parameters: true positive rate (TPR), and false positive rate (FPR)
            - TPR is a synonym for recall
            - FPR is: (FP)/(FP+TN)
            - an ROC curve plots TPR and FPR at different classification thresholds
            - lowering the classification threshold classifies more items as positive, thus increasing both FP and TP
            - the idea is that we evaluate every possible classification threshold and look at the true positive and false                 positive rates at that threshold
            - area under the ROC curve (AUC) has an interesting probabilistic interpretation
            - AUC measures the entire area underneath the entire ROC curve from (0,0) to (1,1)
            - AUC provides an aggregate measure of performance across all possible classification thresholds
            - AUC is the probability that the model ranks a random positive example more highly than a random negative                    example
            - AUC represents the probability that a random positive example is positioned to the right of a random negative               example
            - AUC ranges in value from 0 to 1
            - AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of               what classification threshold chosen
            - AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values
              - AUC is based on relative predictions, so any transformation of the predictions that preserves the relative                   ranking has no effect on AUC. This is not the case with other metrics such as squared error, logloss, or                     prediction bias
        - prediction bias:
          - taking the sum of all of the things we predict and comparing them to the sum of all of the things we observe
          - logistic regression predictions should be unbiased
          - we would like the expected values that we predict, to be equal to the observed values
          - average of predictions should be almost equal to average of observations
            - if they are not, then we say that the model has some bias
            - prediction bias is a quantity that measures how far apart those two averages are.
              - prediction bias = average of predictions - average of labels in data set
              - Note: prediction bias is a different quantity than bias (b in wixi + b)
            - a bias of zero means that the sum of the predictions is equal to the sum of the observations
            - if our model does not have a zero bias, it is a cause for concern. we need to then debug our model. We need to               then slice the data and see what areas the model is not doing a good job of having a zero bias
            - However, having a zero bias is ins itelf not sufficient to tell us that the model is perfect
            - we can look at a more fine grained view of bias by looking at a calibration plot
          - possible root causes of prediction bias are:
            - incomplete feature set
            - noisy data set
            - buggy pipeline
            - biased training sample
            - overly strong regularization
          - we may be tempted to correct prediction bias by post-processing the learned model, i.e., by adding a calibration             layer that adjusts the model's output to reduce the prediction bias
          - however, adding a calibration layer is a bad idea for the following reasons:
            - we are fixing the symptom rather than the cause
            - the system created is a brittle one which needs to be kept up-to-date
          - if possible, avoid calibration layers
          - a good model will usually have near-zero bias. That doesn't mean that a low prediction bias makes a model good
          - logistic regression predicts a value between 0 and 1. Labeled examples are either exactly 0 or exactly 1
          - to determine prediction bias, we need not one but a bucket of examples. That is, prediction bias for logisitic               regression makes sense only when grouping enough examples together to be able to then compare a predicted value             with the observed values.          
          - buckets can be made by either linearly breaking up the target predictions, or by forming quantiles
        
        - Binary classification model:
          - we should normalize features in a multi-feature model. The value of each feature should cover roughly the same               range
          - the following code cell normalizes datasets by converting each raw value to its Z-score. A Z-score is the number             of standard deviations from the mean for a particular raw value
              - train_df_mean = train_df.mean()
              - train_df_std = train_df.std()
              - train_df_norm = (train_df - train_df_mean)/train_df_std
              - train_df_norm.head()
            - consider a feature with mean = 60; and standard deviation  = 10; A raw value of 75 will then have a Z-score =               (75-mean)/standard deviation = +1.5
            - A raw value of 38 will have a Z-score of -2.2
            - in classification problems, the label for every example must be either 0 or 1. 
              - to convert True and False to 1 and 0, call the pandas DataFrame function "astype(float)"
                - train_df_norm["median_house_value_high"] = (train_df["median_house_value"] > threshold).astype(float)
              
   - Regularization: Sparcity
     - sparse feature crosses may lead to over-fitting and substantial increase in RAM requirement which may possibly slow          down runtime
     - we want to regularize in a way which also reduces model size/memory usage
       - in a high-dimensional sparse vector, it would be nice to encourage weights to drop exactly to zero, where possible.
         A weight of exactly zero essentially removes the corresponding feature from the model. Zeroing out the features              will save RAM and may reduce noise (over-fitting) in the model
       - we will zero out some of the weights and therefore, avoiding particular crosses
       - we would like to expicitely zero out weights - also known as L0 regularization
       - L0 regularization will penalize for having a weight that was not zero
         - its hard to optimize as it is not convex (non-convex optimization)
         - it refers to the count-based approach where the count of non-zero coefficient values in a model is penalized
         - count-based approach would turn a convex optimization problem into a non-convex optimization problem
       - Instead, L0 regularization is relaxed to L1 regularization
         - it penalizes the sum of the absolute values of the weights
         - by doing this, the model is still encouraged to be very sparse
       - L2 regularization would also drive the weights to be small, but won't make them exactly zero.
       - In other words, L2 and L1 penalize weights differently:
         - L2 penalizes weight^2
         - L1 penalizes abs(weight)
       - Consequently, L2 and L1 have different derivatives:
         - the derivative of L2 is 2*weight
           - One can think of the derivative of L2 as a force that removes x% of weight everytime. At any rate, L2 does not              normally drive weights to zero
           
         - the derivative of L1 is a constant whose value is independent of weight
           - One can think of the derivative of L1 as a force that subtracts some constant from the weight everytime. Thanks              to absolute values, L1 has a discontinuity at 0, which causes subtaction results that cross 0 to become zeroed              out.
        
         - L1 regularization, penalizing the absolute value of all the weights, turns out to be quite efficient for wide                models
         
       - L2 and L1 regularization comparison:
         - For L2 regularization, increasing the regularization rate decreases the difference between test and training loss
         - In case of L1 regularization, the difference of training and test loss is much lower
         - For both L2 and L1 regularization, increasing the lambda decreases the significant weights
         - In L2 regularization, the individual weights never go down to zero
         - In L1 regularization, the individual weights for a number features become exactly zero
         - For L1 regularization, a lambda of 1 makes all the weights to go to zero (always the case?)
         - For L1 regularization, a lambda > 1 produces even smaller weight values for significant weights (always?)         
         
       - L1 regularization may cause the following kinds of features to get a weight of exactly 0:
         - weakly informative features
         - strongly informative features on different scales
         - informative features strongly correlated with other similarly informative features
       - Nevertheless, L1 regularization tends to reduce the number of features, and decrease the model size
         
            
   - Neural Networks:
      - we would like the model in someway to learn the nonlinearities by themselves without us having to specify them               manually
      - for this we need a model with some additional structure
      - we stick in nonlinearity between the hidden layers, known as nonlinear transformation layer
      - a nonlinear transformation layer is also known as activation function
      - we can pipe each hidden layer node through a nonlinear function
      - the value of each node in a preceding hidden layer is transformed by a nonlinear function before being passed on to         the weighted sums of the next layer
      - nonlinear transformationlayer can go at the output of any of the hidden nodes
      - after adding an activation function, adding layers has more impact
      - stacking nonlinearities on nonlinearities lets us model very complicated relationship between the inputs and the             predicted outputs
      - one common nonlinearity that is used is called ReLU 
      - RELU is rectified linear unit
      - F(x) = max(0,x)
      - ReLU takes in a linear function and chops off the portion which lies in the region <=0 and makes that portion 0
      - ReLU is a simple nonlinear function, which allows us to create nonlinear models
      - ReLU gives state-of-the-art results for wide number of problems
      - the superiority of ReLU is based on empirical findings, probably driven by ReLU having a more useful range of               responsiveness
      - Other nonlinear functions include sigmoid or tanh
      - In fact, any mathematical function can serve as an activation function
      - If sigma is our activation function, then the value of a node in the network is:
         - sigma*(w*x+b)
      
      - A neural network comprises of:
         - a set of nodes, analogous to neurons, organized in layers
         - a set of weights representing the connections between each neural network layer and the layer beneath it
         - a set of biases, one for each node
         - an activation function that transforms the output of each node in a layer. Different layers may have different              activation functions
      
      - this is what deep neural nets do
        - we can create arbitrarily complex neural network
        - they do specially good job at complex data including image data, audio data and video data
      - when we train these neural nets, we are in a non convex optimization, so initialization may matter
      - the method that is used to train neural nets, is a variant of gradient descent, called back propagation
      - back propagation essentially allows us to do gradient descent in this non-convex optimization in a reasonably               efficient manner
      - neural networks are not necessarily always better than feature crosses, but neural networks do offer a flexible             alternative that works well in many cases
    
  start from playground exercises
