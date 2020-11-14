# Machine_Learning (ML)
Machine Learning is the field of study that gives computers the ability to learn without  being explicitly programmed.<br>

ML is divided into two categories:<br>
-Supervised <br>
-Unsupervised<br>

Supervised is where data is labelled and the program learns to predict the output from the input data<br>

This can be broken into: <br>
-Regression: problems here include prediction of a continous valued output e.g housing prices in Nairobi. <br>
-Classification: problems here deal with prediction of discrete number of values e.g is a particlar email spam or not <br>

#### Regression Vs Classification
Regression is used to predict outputs that are continuous. 
The outputs are quantities that can be flexibly determined based on the inputs of the model rather than being confined to a set of possible labels.<br>
For example: Predict the height of a potted plant from the amount of rainfall<br>

Classification
Classification is used to predict a discrete label.
The outputs fall under a finite set of possible outcomes. 
Many situations have only two possible outcomes. This is called binary classification(True/False).<br>
For example: Predict whether it will rain or not<br>

Multi-label classification is when there are multiple possible outcomes. 
It is useful for customer segmentation, image categorization, and sentiment analysis for understanding text. 
To perform these classifications, we use models like Naive Bayes, K-Nearest Neighbors, and SVMs.

Unsupervised Learning is where the program learns the inherent structure of the data based on unlabeled examples.<br>

Clustering is a common unsupervised machine learning approach that finds patterns and structures in unlabeled data by grouping them into clusters. e.g Search engines to group similar objects in one cluster <br>

#### Differences.
Supervised Learning: data is labeled and the program learns to predict the output from the input data.<br>
Unsupervised Learning: data is unlabeled and the program learns to recognize the inherent structure in the input data. <br>

## The Process.
The process of performing Machine Learning often requires many more steps before and after the predictive analytics.<br>
We try to think of the Machine Learning process as:<br>

#### 1.Formulating a Question.<br>
What is it that we want to find out? How will we reach the success criteria that we set?<br>
When we’re thinking about creating a model, we have to narrow down to one measurable, specific task. For example, we might say we want to predict the wait times for customers’ food orders within 2 minutes, so that we can give them an accurate time estimate.

#### 2.Finding and Understanding the Data.<br>
The largest chunk of time in any machine learning process is finding the relevant data to help answer your question, and getting it into the format necessary for performing predictive analysis.<br>
Once you have your data, you want to understand it so that you will know what model to apply and what the outputs will mean. First, you will want to examine the summary statistics:<br>
   -Calculate means and medians to understand the distribution<br>
   -Calculate percentiles.<br>
   -Find correlations that indicate relationships.<br>
You may also want to visualize the data, perhaps using box plots to identify outliers, histograms to show the basic structure of the data, and scatter plots to examine relationships between variables.<br>

#### 3.Cleaning the Data and Feature Engineering.<br>
Real data is messy! Data may have errors. Some columns may be empty. The features we’re interested in might require string manipulation to extract. Cleaning the data refers to the process by which we address missing values and outliers, among other things that may affect our insights.<br>
Feature Engineering refers to the process by which we choose the important features (or columns) to look at, and make the appropriate transformations to prepare our data for our model.<br>
We might try:<br>
   -Normalizing or standardizing the data.<br>
   -Augmenting the data by adding new columns.<br>
   -Removing unnecessary columns.<br>
After we test our model on the data we have, we might go back and reengineer features to see if we get a better result.<br>

#### 4.Choosing a Model.<br>
Once we understand our dataset and know the problem we are trying to solve, we can begin to choose a model that will help us tackle our problem.<br>
If we are attempting to find a continuous output, like predicting the number of minutes someone should wait for their order, we would use a regression algorithm.<br>
If we are attempting to classify an input, like determining if an order will take under 5 minutes or over 10 mins, then we would use a classification algorithm.<br>

#### 5.Tuning and Evaluating.<br>
Each model has a variety of parameters that change how it makes decisions. We can adjust these and compare the chosen evaluation metrics of the different variants to find the most accurate model.<br>

#### 6.Using the Model and Presenting Results.<br>
When you achieve the level of accuracy you want on your training set, you can use the model on the data you actually care about analyzing.<br>
For our example, we can now start inputting new orders. The input could be an order, with features like:<br>
   -the type of item ordered.<br>
   -the quantity.<br>
   -the time of day.<br>
   -the number of employees working.<br>
The output would be how long the order is expected to take. This information could be displayed to users.

## Scikit-Learn.
Scikit-learn is a library in Python that provides many unsupervised and supervised learning algorithms. It’s built upon some of the technologies like NumPy, pandas, and Matplotlib! <br>

The functionality that scikit-learn provides include:
    -Regression, including Linear and Logistic Regression.<br>
    -Classification, including K-Nearest Neighbors.<br>
    -Clustering, including K-Means and K-Means++ .<br>
    -Model selection.<br>
    -Preprocessing, including Min-Max Normalization.<br>
    
### Scikit-Learn Cheatsheet.
#### Linear Regression.<br>
Import and create the model: <br>

```python 
from sklearn.linear_model import LinearRegression.

your_model = LinearRegression().
```

fit: <br>

```python
your_model.fit(x_training_data, y_training_data)
```
.coef_: contains the coefficients<br>
.intercept_: contains the intercept<br>

Predict:<br>

``` python
predictions = your_model.predict(your_x_data)
```
.score(): returns the coefficient of determination R².

#### Naive Bayes
Import and create the model: <br>

```python
from sklearn.naive_bayes import MultinomialNB

your_model = MultinomialNB()
```
Fit:<br>

```python
your_model.fit(x_training_data, y_training_data)
```

Predict:<br>

``` python
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
```
```python
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)

```

#### K-Nearest Neighbors
Import and create the model:<br>

```python
from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()
```
Fit:<br>

```python
your_model.fit(x_training_data, y_training_data)
```
Predict:<br>

```python
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
```
```python
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)
```

#### K-Means
Import and create the model:<br>

```python
from sklearn.cluster import KMeans

your_model = KMeans(n_clusters=4, init='random')
```
n_clusters: number of clusters to form and number of centroids to generate<br>

init: method for initialization<br>

k-means++: K-Means++ [default] <br>

random: K-Means<br>

random_state: the seed used by the random number generator [optional]<br>

Fit:<br>

```python
your_model.fit(x_training_data)
```

Predict:<br>

```python
predictions = your_model.predict(your_x_data)
```
#### Validating the Model
Import and print accuracy, recall, precision, and F1 score:<br>

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(true_labels, guesses))

print(recall_score(true_labels, guesses))

print(precision_score(true_labels, guesses))

print(f1_score(true_labels, guesses))
``` 

Import and print the confusion matrix:<br>

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(true_labels, guesses))
```
#### Training Sets and Test Sets 
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
```
train_size: the proportion of the dataset to include in the train split<br>

test_size: the proportion of the dataset to include in the test split <br>

random_state: the seed used by the random number generator [optional] <br>

### Linear Regression
Representing Points, distance.<br>
Three different ways to define the distance between two points:<br>
-Euclidean Distance<br>
Euclidean Distance is the most commonly used distance formula. 
To find the Euclidean distance between two points, we first calculate the squared distance between each dimension. 
If we add up all of these squared differences and take the square root, we’ve computed the Euclidean distance.
 
-Manhattan Distance<br>
Manhattan Distance is extremely similar to Euclidean distance. 
Rather than summing the squared difference between each dimension, 
we instead sum the absolute value of the difference between each dimension

-Hamming Distance<br>
Instead of finding the difference of each dimension, 
Hamming distance only cares about whether the dimensions are exactly equal. 
When finding the Hamming distance between two points, add one for every dimension that has different values.
Hamming distance is used in spell checking algorithms.<br>

SciPy Distances.<br>
Implementation of the above distances using scipy python library.<br>
-Euclidean Distance .euclidean()<br>
-Manhattan Distance .cityblock()<br>
-Hamming Distance .hamming()<br>
scipy implementation of Hamming distance will always return a number between 0 an 1.

## Data Manipulation with Numpy
### Numpy Arrays
a. Numpy Arrays <br>
NumPy arrays are basically just Python lists with added features. You can easily convert a Python list to a Numpy array using the np.array function which takes in a Python list as its required argument. The function has quite a few keyword arguments, but the main one to know is dtype. 
The dtype keyword argument takes in a NumPy type and manually casts the array to the specified type.<br>
Example:<br>
The code below is an example usage of np.array to create a 2-D matrix. Note that the array is manually cast to np.float32.<br>

```python 
import numpy as np

arr = np.array([[0, 1, 2], [3, 4, 5]],
               dtype=np.float32)
print(repr(arr))

#output array([[0., 1., 2.], [3., 4., 5.]], dtype=float32)
```

When the elements of a NumPy array are mixed types, then the array's type will be upcast to the highest level type. Meaning that if an array input has mixed int and float elements, all the integers will be cast to their floating-point equivalents. 
If an array is mixed with int, float, and string elements, everything is cast to strings.<br>
Example of np.array upcasting. Both integers are cast to their floating-point equivalents.<br>
```python
import numpy as np

arr = np.array([0, 0.1, 2])
print(repr(arr))

#output array([0. , 0.1, 2. ])
```

b. Copying<br>
Similar to Python lists, when we make a reference to a NumPy array it doesn't create a different array. Therefore, if we change a value using the reference variable, it changes the original array as well. We get around this by using an array's inherent copy function. The function has no required arguments, and it returns the copied array.<br>
Example below, c is a reference to a while d is a copy. Therefore, changing c leads to the same change in a, while changing d does not change the value of b.<br>
```python
import numpy as np
a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))

d = b.copy()
d[0] = 6
print('Array b: {}'.format(repr(b)))

#output Array a: array([0, 1]),Array a: array([5, 1]),Array b: array([9, 8])
```

c. Casting<br>
We cast NumPy arrays through their inherent astype function. The function's required argument is the new type for the array.
It returns the array cast to the new type.<br>
The  example below is on casting using the astype function. The dtype property returns the type of an array.<br>
```python
arr = np.array([0, 1, 2])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)

#output int64 float32
```
d. NaN<br>
When we don't want a NumPy array to contain a value at a particular index, we can use np.nan to act as a placeholder.
A common usage for np.nan is as a filler value for incomplete data.<br>
Example usage of np.nan. Note that np.nan cannot take on an integer type.<br>
```python
arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

# Will result in a ValueError
np.array([np.nan, 1, 2], dtype=np.int32)

#other ouput array([nan,  1.,  2.]) array(['nan', 'abc'], dtype='<U32')
```

e. Infinity<br>
To represent infinity in NumPy, we use the np.inf special value. We can also represent negative infinity with -np.inf.<br>
Example usage of np.inf. Note that np.inf cannot take on an integer type.<br>

```python
print(np.inf > 1000000)

arr = np.array([np.inf, 5])
print(repr(arr))

arr = np.array([-np.inf, 1])
print(repr(arr))

# Will result in an OverflowError
np.array([np.inf, 3], dtype=np.int32)

#other output True, array([inf,  5.]), array([-inf,   1.])
```
More illustrations of the above can be found [here](https://github.com/veldakarimi/Machine_Learning/blob/ca21216a2b73282fe7ac71e3f3c7a0bce738be2a/numpyarrays.py#L1-L14)<br>
### Numpy Basics
Perform basic operations to create and modify NumPy arrays.
a. Ranged Data<br>
np.array  is used to create an array which is equal to hardcoding.<br>
NumPy provides an option to create ranged data arrays using np.arange. 
The function acts very similar to the range function in Python, and will always return a 1-D array.
Example:<br>
```python
arr = np.arange(5)
print(repr(arr))

arr = np.arange(5.1)
print(repr(arr))

arr = np.arange(-1, 4)
print(repr(arr))

arr = np.arange(-1.5, 4, 2)
print(repr(arr))

#output
#array([0, 1, 2, 3, 4])
#array([0., 1., 2., 3., 4., 5.])
#array([-1,  0,  1,  2,  3])
#array([-1.5,  0.5,  2.5])
```
The output of np.arange is specified as follows:<br>
-If only a single number, n, is passed in as an argument, np.arange will return an array with all the integers in the range [0,n] Note: the lower end is inclusive while the upper end is exclusive.<br>
-For two arguments, m and n, np.arange will return an array with all the integers in the range [m, n].<br>
-For three arguments, m, n, and s, np.arange will return an array with the integers in the range [m, n] using a step size of s.<br>
-Like np.array, np.arange performs upcasting. It also has the dtype keyword argument to manually cast the array.<br>

To specify the number of elements in the returned array, rather than the step size, we can use the np.linspace function.<br>
This function takes in a required first two arguments, for the start and end of the range, respectively.<br>
The end of the range is inclusive for np.linspace, unless the keyword argument endpoint is set to False.<br>
To specify the number of elements, we set the num keyword argument (its default value is 50).
Example:<br>
```python
arr = np.linspace(5, 11, num=4)
print(repr(arr))

arr = np.linspace(5, 11, num=4, endpoint=False)
print(repr(arr))

arr = np.linspace(5, 11, num=4, dtype=np.int32)
print(repr(arr))

#output
#array([ 5.,  7.,  9., 11.])
#array([5. , 6.5, 8. , 9.5])
#array([ 5,  7,  9, 11], dtype=int32)
```
b. Reshaping data<br>
We use np.reshape to reshape data.<br>
It takes in an array and a new shape as required arguments. The new shape must exactly contain all the elements from the input array. E.g, we could reshape an array with 12 elements to (4, 3), but we can't reshape it to (4, 4).<br>
We are allowed to use the special value of -1 in at most one dimension of the new shape. The dimension with -1 will take on the value necessary to allow the new shape to contain all the elements of the array.
Example:<br>
```python
arr = np.arange(8)

reshaped_arr = np.reshape(arr, (2, 4))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

reshaped_arr = np.reshape(arr, (-1, 2, 2))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))
#output array([[0, 1, 2, 3], [4, 5, 6, 7]]) New shape: (2, 4) array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]) New shape: (2, 2, 2)
```
While the np.reshape function can perform any reshaping utilities we need, NumPy provides an inherent function for flattening an array.<br>
Flattening an array reshapes it into a 1D array. Since we need to flatten data quite often, it is a useful function.<br>
Example:<br>
```python
arr = np.arange(8)
arr = np.reshape(arr, (2, 4))
flattened = arr.flatten()
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(flattened))
print('flattened shape: {}'.format(flattened.shape))

#output 
#array([[0, 1, 2, 3],
       #[4, 5, 6, 7]])
#arr shape: (2, 4)
#array([0, 1, 2, 3, 4, 5, 6, 7])
#flattened shape: (8,)
```
c. Transposing<br>
The general meaning of transposing is to change places or context.<br>
Perharps we have data that's supposed to be in a particular format, but some new data we get is rearranged. We can transpose the data, using the np.transpose function, to convert it to the proper format.<br>
The code below shows an example usage of the np.transpose function. The matrix rows become columns after the transpose.
```python
arr = np.arange(8)
arr = np.reshape(arr, (4, 2))
transposed = np.transpose(arr)
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(transposed))
print('transposed shape: {}'.format(transposed.shape))

#output
#array([[0, 1],
       #[2, 3],
       #[4, 5],
       #[6, 7]])
#arr shape: (4, 2)
#array([[0, 2, 4, 6],
       #[1, 3, 5, 7]])
#transposed shape: (2, 4)
```
The function takes in a required first argument, which will be the array we want to transpose. It also has a single keyword argument called <b>axes</b>, which represents the new permutation of the dimensions.<br>
The permutation <i>is a tuple/list of integers, with the same length as the number of dimensions in the array</i>. It tells us where to switch up the dimensions. For example, if the permutation had 3 at index 1, it means the old third dimension of the data becomes the new second dimension (since index 1 represents the second dimension).<br>
The code below shows an example usage of the np.transpose function with the axes keyword argument. The shape property gives us the shape of an array.<br>
```python
arr = np.arange(24)
arr = np.reshape(arr, (3, 4, 2))
transposed = np.transpose(arr, axes=(1, 2, 0))
print('arr shape: {}'.format(arr.shape))
print('transposed shape: {}'.format(transposed.shape))

#output
#arr shape: (3, 4, 2)
#transposed shape: (4, 2, 3)
```
In this example, the old first dimension became the new third dimension, the old second dimension became the new first dimension, and the old third dimension became the new second dimension. The default value for axes is a dimension reversal (e.g. for 3-D data the default axes value is [2, 1, 0]).

d. Zeros and ones
Sometimes, we need to create arrays filled solely with 0 or 1.e.g binary data, we may need to create dummy datasets of strictly one label. For creating these arrays, NumPy provides the functions np.zeros and np.ones. They both take in the same arguments, which includes just one required argument, the array shape.<br>
The functions also allow for manual casting using the dtype keyword argument.<br>
The code below shows example usages of np.zeros and np.ones.<br>
```python
arr = np.zeros(4)
print(repr(arr))

arr = np.ones((2, 3))
print(repr(arr))

arr = np.ones((2, 3), dtype=np.int32)
print(repr(arr))

#output
#array([0., 0., 0., 0.])
#array([[1., 1., 1.],
       #[1., 1., 1.]])
#array([[1, 1, 1],
       #[1, 1, 1]], dtype=int32)
```
If we want to create an array of 0's or 1's with the same shape as another array, we can use np.zeros_like and np.ones_like.<br>
The code below shows example usages of np.zeros_like and np.ones_like.<br>
```python
arr = np.array([[1, 2], [3, 4]])
print(repr(np.zeros_like(arr)))

arr = np.array([[0., 1.], [1.2, 4.]])
print(repr(np.ones_like(arr)))
print(repr(np.ones_like(arr, dtype=np.int32)))
#output
#array([[0, 0],
       #[0, 0]])
#array([[1., 1.],
       #[1., 1.]])
#array([[1, 1],
       #[1, 1]], dtype=int32)
```
More examples on Numpy basics [here](https://github.com/veldakarimi/Machine_Learning/blob/6119cb356382d3e22c15eb1115d42f0f4cb6a5da/numpybasics.py#L1-L11)

### Math
a. Arithmetic <br>
One of the main purposes of NumPy is to perform multi-dimensional arithmetic.<br>
Using NumPy arrays, we can apply arithmetic to each element with a single operation.<br>
The code below shows multi-dimensional arithmetic with NumPy.<br>
```python
arr = np.array([[1, 2], [3, 4]])
# Add 1 to element values
print(repr(arr + 1))
# Subtract element values by 1.2
print(repr(arr - 1.2))
# Double element values
print(repr(arr * 2))
# Halve element values
print(repr(arr / 2))
# Integer division (half)
print(repr(arr // 2))
# Square element values
print(repr(arr**2))
# Square root element values
print(repr(arr**0.5))

#output
array([[2, 3],
       [4, 5]])
array([[-0.2,  0.8],
       [ 1.8,  2.8]])
array([[2, 4],
       [6, 8]])
array([[0.5, 1. ],
       [1.5, 2. ]])
array([[0, 1],
       [1, 2]])
array([[ 1,  4],
       [ 9, 16]])
array([[1.        , 1.41421356],
       [1.73205081, 2.        ]])
```
Using NumPy arithmetic, we can easily modify large amounts of numeric data with only a few operations. e.g, we could convert a dataset of Fahrenheit temperatures to their equivalent Celsius form.<br>
The code below converts Fahrenheit to Celsius in NumPy.<br>
```python
def f2c(temps):
  return (5/9)*(temps-32)

fahrenheits = np.array([32, -4, 14, -40])
celsius = f2c(fahrenheits)
print('Celsius: {}'.format(repr(celsius)))

#output
Celsius: array([  0., -20., -10., -40.])
```
NB:Performing arithmetic on NumPy arrays<b> does not change the original array</b>, and instead produces a new array that is the result of the arithmetic operation.

b. Non-linear functions<br>
NumPy also allows you to use non-linear functions such as exponentials and logarithms.<br>
The function np.exp performs a base e exponential on an array, while the function np.exp2 performs a base 2 exponential. 
Likewise, np.log, np.log2, and np.log10 all perform logarithms on an input array, using base e, base 2, and base 10, respectively.<br>
The code below shows various exponentials and logarithms with NumPy. Note that np.e and np.pi represent the mathematical constants e and π, respectively.<br>
```python
arr = np.array([[1, 2], [3, 4]])
# Raised to power of e
print(repr(np.exp(arr)))
# Raised to power of 2
print(repr(np.exp2(arr)))

arr2 = np.array([[1, 10], [np.e, np.pi]])
# Natural logarithm
print(repr(np.log(arr2)))
# Base 10 logarithm
print(repr(np.log10(arr2)))

#output
array([[ 2.71828183,  7.3890561 ],
       [20.08553692, 54.59815003]])
array([[ 2.,  4.],
       [ 8., 16.]])
array([[0.        , 2.30258509],
       [1.        , 1.14472989]])
array([[0.        , 1.        ],
       [0.43429448, 0.49714987]])
```
To use a regular power operation with any base, we use np.power. The first argument to the function is the base, while the second is the power. <br>
If the base or power is an array rather than a single number, the operation is applied to every element in the array.<br>
Example of using np.power:<br>
```python
arr = np.array([[1, 2], [3, 4]])
# Raise 3 to power of each number in arr
print(repr(np.power(3, arr)))
arr2 = np.array([[10.2, 4], [3, 5]])
# Raise arr2 to power of each number in arr
print(repr(np.power(arr2, arr)))

#output
array([[ 3,  9],
       [27, 81]])
array([[ 10.2,  16. ],
       [ 27. , 625. ]])
```
NumPy has various other mathematical functions, which are listed [here](https://numpy.org/doc/stable/reference/routines.math.html)

c. Matrix multiplication<br>
NumPy arrays are basically vectors and matrices, it makes sense that there are functions for dot products and matrix multiplication.The main function to use is np.matmul, which takes two vector/matrix arrays as input and produces a dot product or matrix multiplication.<br>
Nb: that the dimensions of the two input matrices must be valid for a matrix multiplication. Specifically, the second dimension of the first matrix must equal the first dimension of the second matrix, otherwise np.matmul will result in a ValueError.<br>
The code below shows various examples of matrix multiplication. When both inputs are 1-D, the output is the dot product.
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([-3, 0, 10])
print(np.matmul(arr1, arr2))

arr3 = np.array([[1, 2], [3, 4], [5, 6]])
arr4 = np.array([[-1, 0, 1], [3, 2, -4]])
print(repr(np.matmul(arr3, arr4)))
print(repr(np.matmul(arr4, arr3)))
# This will result in ValueError
print(repr(np.matmul(arr3, arr3)))

#output
27
array([[  5,   4,  -7],
       [  9,   8, -13],
       [ 13,  12, -19]])
array([[  4,   4],
       [-11, -10]])
Traceback
```







=======
# Machine_Learning (ML)
Machine Learning is the field of study that gives computers the ability to learn without  being explicitly programmed.<br>

ML is divided into two categories:<br>
-Supervised <br>
-Unsupervised<br>

Supervised is where data is labelled and the program learns to predict the output from the input data<br>

This can be broken into: <br>
-Regression: problems here include prediction of a continous valued output e.g housing prices in Nairobi. <br>
-Classification: problems here deal with prediction of discrete number of values e.g is a particlar email spam or not <br>

#### Regression Vs Classification
Regression is used to predict outputs that are continuous. 
The outputs are quantities that can be flexibly determined based on the inputs of the model rather than being confined to a set of possible labels.<br>
For example: Predict the height of a potted plant from the amount of rainfall<br>

Classification
Classification is used to predict a discrete label.
The outputs fall under a finite set of possible outcomes. 
Many situations have only two possible outcomes. This is called binary classification(True/False).<br>
For example: Predict whether it will rain or not<br>

Multi-label classification is when there are multiple possible outcomes. 
It is useful for customer segmentation, image categorization, and sentiment analysis for understanding text. 
To perform these classifications, we use models like Naive Bayes, K-Nearest Neighbors, and SVMs.

Unsupervised Learning is where the program learns the inherent structure of the data based on unlabeled examples.<br>

Clustering is a common unsupervised machine learning approach that finds patterns and structures in unlabeled data by grouping them into clusters. e.g Search engines to group similar objects in one cluster <br>

#### Differences.
Supervised Learning: data is labeled and the program learns to predict the output from the input data.<br>
Unsupervised Learning: data is unlabeled and the program learns to recognize the inherent structure in the input data. <br>

## The Process.
The process of performing Machine Learning often requires many more steps before and after the predictive analytics.<br>
We try to think of the Machine Learning process as:<br>

#### 1.Formulating a Question.<br>
What is it that we want to find out? How will we reach the success criteria that we set?<br>
When we’re thinking about creating a model, we have to narrow down to one measurable, specific task. For example, we might say we want to predict the wait times for customers’ food orders within 2 minutes, so that we can give them an accurate time estimate.

#### 2.Finding and Understanding the Data.<br>
The largest chunk of time in any machine learning process is finding the relevant data to help answer your question, and getting it into the format necessary for performing predictive analysis.<br>
Once you have your data, you want to understand it so that you will know what model to apply and what the outputs will mean. First, you will want to examine the summary statistics:<br>
   -Calculate means and medians to understand the distribution<br>
   -Calculate percentiles.<br>
   -Find correlations that indicate relationships.<br>
You may also want to visualize the data, perhaps using box plots to identify outliers, histograms to show the basic structure of the data, and scatter plots to examine relationships between variables.<br>

#### 3.Cleaning the Data and Feature Engineering.<br>
Real data is messy! Data may have errors. Some columns may be empty. The features we’re interested in might require string manipulation to extract. Cleaning the data refers to the process by which we address missing values and outliers, among other things that may affect our insights.<br>
Feature Engineering refers to the process by which we choose the important features (or columns) to look at, and make the appropriate transformations to prepare our data for our model.<br>
We might try:<br>
   -Normalizing or standardizing the data.<br>
   -Augmenting the data by adding new columns.<br>
   -Removing unnecessary columns.<br>
After we test our model on the data we have, we might go back and reengineer features to see if we get a better result.<br>

#### 4.Choosing a Model.<br>
Once we understand our dataset and know the problem we are trying to solve, we can begin to choose a model that will help us tackle our problem.<br>
If we are attempting to find a continuous output, like predicting the number of minutes someone should wait for their order, we would use a regression algorithm.<br>
If we are attempting to classify an input, like determining if an order will take under 5 minutes or over 10 mins, then we would use a classification algorithm.<br>

#### 5.Tuning and Evaluating.<br>
Each model has a variety of parameters that change how it makes decisions. We can adjust these and compare the chosen evaluation metrics of the different variants to find the most accurate model.<br>

#### 6.Using the Model and Presenting Results.<br>
When you achieve the level of accuracy you want on your training set, you can use the model on the data you actually care about analyzing.<br>
For our example, we can now start inputting new orders. The input could be an order, with features like:<br>
   -the type of item ordered.<br>
   -the quantity.<br>
   -the time of day.<br>
   -the number of employees working.<br>
The output would be how long the order is expected to take. This information could be displayed to users.

## Scikit-Learn.
Scikit-learn is a library in Python that provides many unsupervised and supervised learning algorithms. It’s built upon some of the technologies like NumPy, pandas, and Matplotlib! <br>

The functionality that scikit-learn provides include:
    -Regression, including Linear and Logistic Regression.<br>
    -Classification, including K-Nearest Neighbors.<br>
    -Clustering, including K-Means and K-Means++ .<br>
    -Model selection.<br>
    -Preprocessing, including Min-Max Normalization.<br>
    
### Scikit-Learn Cheatsheet.
#### Linear Regression.<br>
Import and create the model: <br>

```python 
from sklearn.linear_model import LinearRegression.

your_model = LinearRegression().
```

fit: <br>

```python
your_model.fit(x_training_data, y_training_data)
```
.coef_: contains the coefficients<br>
.intercept_: contains the intercept<br>

Predict:<br>

``` python
predictions = your_model.predict(your_x_data)
```
.score(): returns the coefficient of determination R².

#### Naive Bayes
Import and create the model: <br>

```python
from sklearn.naive_bayes import MultinomialNB

your_model = MultinomialNB()
```
Fit:<br>

```python
your_model.fit(x_training_data, y_training_data)
```

Predict:<br>

``` python
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
```
```python
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)

```

#### K-Nearest Neighbors
Import and create the model:<br>

```python
from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()
```
Fit:<br>

```python
your_model.fit(x_training_data, y_training_data)
```
Predict:<br>

```python
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
```
```python
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)
```

#### K-Means
Import and create the model:<br>

```python
from sklearn.cluster import KMeans

your_model = KMeans(n_clusters=4, init='random')
```
n_clusters: number of clusters to form and number of centroids to generate<br>

init: method for initialization<br>

k-means++: K-Means++ [default] <br>

random: K-Means<br>

random_state: the seed used by the random number generator [optional]<br>

Fit:<br>

```python
your_model.fit(x_training_data)
```

Predict:<br>

```python
predictions = your_model.predict(your_x_data)
```
#### Validating the Model
Import and print accuracy, recall, precision, and F1 score:<br>

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(true_labels, guesses))

print(recall_score(true_labels, guesses))

print(precision_score(true_labels, guesses))

print(f1_score(true_labels, guesses))
``` 

Import and print the confusion matrix:<br>

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(true_labels, guesses))
```
#### Training Sets and Test Sets 
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
```
train_size: the proportion of the dataset to include in the train split<br>

test_size: the proportion of the dataset to include in the test split <br>

random_state: the seed used by the random number generator [optional] <br>

### Linear Regression
Representing Points, distance.<br>
Three different ways to define the distance between two points:<br>
-Euclidean Distance<br>
Euclidean Distance is the most commonly used distance formula. 
To find the Euclidean distance between two points, we first calculate the squared distance between each dimension. 
If we add up all of these squared differences and take the square root, we’ve computed the Euclidean distance.
 
-Manhattan Distance<br>
Manhattan Distance is extremely similar to Euclidean distance. 
Rather than summing the squared difference between each dimension, 
we instead sum the absolute value of the difference between each dimension

-Hamming Distance<br>
Instead of finding the difference of each dimension, 
Hamming distance only cares about whether the dimensions are exactly equal. 
When finding the Hamming distance between two points, add one for every dimension that has different values.
Hamming distance is used in spell checking algorithms.<br>

SciPy Distances.<br>
Implementation of the above distances using scipy python library.<br>
-Euclidean Distance .euclidean()<br>
-Manhattan Distance .cityblock()<br>
-Hamming Distance .hamming()<br>
scipy implementation of Hamming distance will always return a number between 0 an 1.

## Data Manipulation with Numpy
### Numpy Arrays
a. Numpy Arrays <br>
NumPy arrays are basically just Python lists with added features. You can easily convert a Python list to a Numpy array using the np.array function which takes in a Python list as its required argument. The function has quite a few keyword arguments, but the main one to know is dtype. 
The dtype keyword argument takes in a NumPy type and manually casts the array to the specified type.<br>
Example:<br>
The code below is an example usage of np.array to create a 2-D matrix. Note that the array is manually cast to np.float32.<br>

```python 
import numpy as np

arr = np.array([[0, 1, 2], [3, 4, 5]],
               dtype=np.float32)
print(repr(arr))

#output array([[0., 1., 2.], [3., 4., 5.]], dtype=float32)
```

When the elements of a NumPy array are mixed types, then the array's type will be upcast to the highest level type. Meaning that if an array input has mixed int and float elements, all the integers will be cast to their floating-point equivalents. 
If an array is mixed with int, float, and string elements, everything is cast to strings.<br>
Example of np.array upcasting. Both integers are cast to their floating-point equivalents.<br>
```python
import numpy as np

arr = np.array([0, 0.1, 2])
print(repr(arr))

#output array([0. , 0.1, 2. ])
```

b. Copying<br>
Similar to Python lists, when we make a reference to a NumPy array it doesn't create a different array. Therefore, if we change a value using the reference variable, it changes the original array as well. We get around this by using an array's inherent copy function. The function has no required arguments, and it returns the copied array.<br>
Example below, c is a reference to a while d is a copy. Therefore, changing c leads to the same change in a, while changing d does not change the value of b.<br>
```python
import numpy as np
a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))

d = b.copy()
d[0] = 6
print('Array b: {}'.format(repr(b)))

#output Array a: array([0, 1]),Array a: array([5, 1]),Array b: array([9, 8])
```

c. Casting<br>
We cast NumPy arrays through their inherent astype function. The function's required argument is the new type for the array.
It returns the array cast to the new type.<br>
The  example below is on casting using the astype function. The dtype property returns the type of an array.<br>
```python
arr = np.array([0, 1, 2])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)

#output int64 float32
```
d. NaN<br>
When we don't want a NumPy array to contain a value at a particular index, we can use np.nan to act as a placeholder.
A common usage for np.nan is as a filler value for incomplete data.<br>
Example usage of np.nan. Note that np.nan cannot take on an integer type.<br>
```python
arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

# Will result in a ValueError
np.array([np.nan, 1, 2], dtype=np.int32)

#other ouput array([nan,  1.,  2.]) array(['nan', 'abc'], dtype='<U32')
```

e. Infinity<br>
To represent infinity in NumPy, we use the np.inf special value. We can also represent negative infinity with -np.inf.<br>
Example usage of np.inf. Note that np.inf cannot take on an integer type.<br>

```python
print(np.inf > 1000000)

arr = np.array([np.inf, 5])
print(repr(arr))

arr = np.array([-np.inf, 1])
print(repr(arr))

# Will result in an OverflowError
np.array([np.inf, 3], dtype=np.int32)

#other output True, array([inf,  5.]), array([-inf,   1.])
```
More illustrations of the above can be found [here](https://github.com/veldakarimi/Machine_Learning/blob/ca21216a2b73282fe7ac71e3f3c7a0bce738be2a/numpyarrays.py#L1-L14)<br>
### Numpy Basics
Perform basic operations to create and modify NumPy arrays.
a. Ranged Data<br>
np.array  is used to create an array which is equal to hardcoding.<br>
NumPy provides an option to create ranged data arrays using np.arange. 
The function acts very similar to the range function in Python, and will always return a 1-D array.
Example:<br>
```python
arr = np.arange(5)
print(repr(arr))

arr = np.arange(5.1)
print(repr(arr))

arr = np.arange(-1, 4)
print(repr(arr))

arr = np.arange(-1.5, 4, 2)
print(repr(arr))

#output
#array([0, 1, 2, 3, 4])
#array([0., 1., 2., 3., 4., 5.])
#array([-1,  0,  1,  2,  3])
#array([-1.5,  0.5,  2.5])
```
The output of np.arange is specified as follows:<br>
-If only a single number, n, is passed in as an argument, np.arange will return an array with all the integers in the range [0,n] Note: the lower end is inclusive while the upper end is exclusive.<br>
-For two arguments, m and n, np.arange will return an array with all the integers in the range [m, n].<br>
-For three arguments, m, n, and s, np.arange will return an array with the integers in the range [m, n] using a step size of s.<br>
-Like np.array, np.arange performs upcasting. It also has the dtype keyword argument to manually cast the array.<br>

To specify the number of elements in the returned array, rather than the step size, we can use the np.linspace function.<br>
This function takes in a required first two arguments, for the start and end of the range, respectively.<br>
The end of the range is inclusive for np.linspace, unless the keyword argument endpoint is set to False.<br>
To specify the number of elements, we set the num keyword argument (its default value is 50).
Example:<br>
```python
arr = np.linspace(5, 11, num=4)
print(repr(arr))

arr = np.linspace(5, 11, num=4, endpoint=False)
print(repr(arr))

arr = np.linspace(5, 11, num=4, dtype=np.int32)
print(repr(arr))

#output
#array([ 5.,  7.,  9., 11.])
#array([5. , 6.5, 8. , 9.5])
#array([ 5,  7,  9, 11], dtype=int32)
```
b. Reshaping data<br>
We use np.reshape to reshape data.<br>
It takes in an array and a new shape as required arguments. The new shape must exactly contain all the elements from the input array. E.g, we could reshape an array with 12 elements to (4, 3), but we can't reshape it to (4, 4).<br>
We are allowed to use the special value of -1 in at most one dimension of the new shape. The dimension with -1 will take on the value necessary to allow the new shape to contain all the elements of the array.
Example:<br>
```python
arr = np.arange(8)

reshaped_arr = np.reshape(arr, (2, 4))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

reshaped_arr = np.reshape(arr, (-1, 2, 2))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))
#output array([[0, 1, 2, 3], [4, 5, 6, 7]]) New shape: (2, 4) array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]) New shape: (2, 2, 2)
```
While the np.reshape function can perform any reshaping utilities we need, NumPy provides an inherent function for flattening an array.<br>
Flattening an array reshapes it into a 1D array. Since we need to flatten data quite often, it is a useful function.<br>
Example:<br>
```python
arr = np.arange(8)
arr = np.reshape(arr, (2, 4))
flattened = arr.flatten()
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(flattened))
print('flattened shape: {}'.format(flattened.shape))

#output 
#array([[0, 1, 2, 3],
       #[4, 5, 6, 7]])
#arr shape: (2, 4)
#array([0, 1, 2, 3, 4, 5, 6, 7])
#flattened shape: (8,)
```
c. Transposing<br>
The general meaning of transposing is to change places or context.<br>
Perharps we have data that's supposed to be in a particular format, but some new data we get is rearranged. We can transpose the data, using the np.transpose function, to convert it to the proper format.<br>
The code below shows an example usage of the np.transpose function. The matrix rows become columns after the transpose.
```python
arr = np.arange(8)
arr = np.reshape(arr, (4, 2))
transposed = np.transpose(arr)
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(transposed))
print('transposed shape: {}'.format(transposed.shape))

#output
#array([[0, 1],
       #[2, 3],
       #[4, 5],
       #[6, 7]])
#arr shape: (4, 2)
#array([[0, 2, 4, 6],
       #[1, 3, 5, 7]])
#transposed shape: (2, 4)
```
The function takes in a required first argument, which will be the array we want to transpose. It also has a single keyword argument called <b>axes</b>, which represents the new permutation of the dimensions.<br>
The permutation <i>is a tuple/list of integers, with the same length as the number of dimensions in the array</i>. It tells us where to switch up the dimensions. For example, if the permutation had 3 at index 1, it means the old third dimension of the data becomes the new second dimension (since index 1 represents the second dimension).<br>
The code below shows an example usage of the np.transpose function with the axes keyword argument. The shape property gives us the shape of an array.<br>
```python
arr = np.arange(24)
arr = np.reshape(arr, (3, 4, 2))
transposed = np.transpose(arr, axes=(1, 2, 0))
print('arr shape: {}'.format(arr.shape))
print('transposed shape: {}'.format(transposed.shape))

#output
#arr shape: (3, 4, 2)
#transposed shape: (4, 2, 3)
```
In this example, the old first dimension became the new third dimension, the old second dimension became the new first dimension, and the old third dimension became the new second dimension. The default value for axes is a dimension reversal (e.g. for 3-D data the default axes value is [2, 1, 0]).

d. Zeros and ones
Sometimes, we need to create arrays filled solely with 0 or 1.e.g binary data, we may need to create dummy datasets of strictly one label. For creating these arrays, NumPy provides the functions np.zeros and np.ones. They both take in the same arguments, which includes just one required argument, the array shape.<br>
The functions also allow for manual casting using the dtype keyword argument.<br>
The code below shows example usages of np.zeros and np.ones.<br>
```python
arr = np.zeros(4)
print(repr(arr))

arr = np.ones((2, 3))
print(repr(arr))

arr = np.ones((2, 3), dtype=np.int32)
print(repr(arr))

#output
#array([0., 0., 0., 0.])
#array([[1., 1., 1.],
       #[1., 1., 1.]])
#array([[1, 1, 1],
       #[1, 1, 1]], dtype=int32)
```
If we want to create an array of 0's or 1's with the same shape as another array, we can use np.zeros_like and np.ones_like.<br>
The code below shows example usages of np.zeros_like and np.ones_like.<br>
```python
arr = np.array([[1, 2], [3, 4]])
print(repr(np.zeros_like(arr)))

arr = np.array([[0., 1.], [1.2, 4.]])
print(repr(np.ones_like(arr)))
print(repr(np.ones_like(arr, dtype=np.int32)))
#output
#array([[0, 0],
       #[0, 0]])
#array([[1., 1.],
       #[1., 1.]])
#array([[1, 1],
       #[1, 1]], dtype=int32)
```
More examples on Numpy basics [here](https://github.com/veldakarimi/Machine_Learning/blob/6119cb356382d3e22c15eb1115d42f0f4cb6a5da/numpybasics.py#L1-L11)

### Math
a. Arithmetic <br>
One of the main purposes of NumPy is to perform multi-dimensional arithmetic.<br>
Using NumPy arrays, we can apply arithmetic to each element with a single operation.<br>
The code below shows multi-dimensional arithmetic with NumPy.<br>
```python
arr = np.array([[1, 2], [3, 4]])
# Add 1 to element values
print(repr(arr + 1))
# Subtract element values by 1.2
print(repr(arr - 1.2))
# Double element values
print(repr(arr * 2))
# Halve element values
print(repr(arr / 2))
# Integer division (half)
print(repr(arr // 2))
# Square element values
print(repr(arr**2))
# Square root element values
print(repr(arr**0.5))

#output
array([[2, 3],
       [4, 5]])
array([[-0.2,  0.8],
       [ 1.8,  2.8]])
array([[2, 4],
       [6, 8]])
array([[0.5, 1. ],
       [1.5, 2. ]])
array([[0, 1],
       [1, 2]])
array([[ 1,  4],
       [ 9, 16]])
array([[1.        , 1.41421356],
       [1.73205081, 2.        ]])
```
Using NumPy arithmetic, we can easily modify large amounts of numeric data with only a few operations. e.g, we could convert a dataset of Fahrenheit temperatures to their equivalent Celsius form.<br>
The code below converts Fahrenheit to Celsius in NumPy.<br>
```python
def f2c(temps):
  return (5/9)*(temps-32)

fahrenheits = np.array([32, -4, 14, -40])
celsius = f2c(fahrenheits)
print('Celsius: {}'.format(repr(celsius)))

#output
Celsius: array([  0., -20., -10., -40.])
```
NB:Performing arithmetic on NumPy arrays<b> does not change the original array</b>, and instead produces a new array that is the result of the arithmetic operation.

b. Non-linear functions<br>
NumPy also allows you to use non-linear functions such as exponentials and logarithms.<br>
The function np.exp performs a base e exponential on an array, while the function np.exp2 performs a base 2 exponential. 
Likewise, np.log, np.log2, and np.log10 all perform logarithms on an input array, using base e, base 2, and base 10, respectively.<br>
The code below shows various exponentials and logarithms with NumPy. Note that np.e and np.pi represent the mathematical constants e and π, respectively.<br>
```python
arr = np.array([[1, 2], [3, 4]])
# Raised to power of e
print(repr(np.exp(arr)))
# Raised to power of 2
print(repr(np.exp2(arr)))

arr2 = np.array([[1, 10], [np.e, np.pi]])
# Natural logarithm
print(repr(np.log(arr2)))
# Base 10 logarithm
print(repr(np.log10(arr2)))

#output
array([[ 2.71828183,  7.3890561 ],
       [20.08553692, 54.59815003]])
array([[ 2.,  4.],
       [ 8., 16.]])
array([[0.        , 2.30258509],
       [1.        , 1.14472989]])
array([[0.        , 1.        ],
       [0.43429448, 0.49714987]])
```
To use a regular power operation with any base, we use np.power. The first argument to the function is the base, while the second is the power. <br>
If the base or power is an array rather than a single number, the operation is applied to every element in the array.<br>
Example of using np.power:<br>
```python
arr = np.array([[1, 2], [3, 4]])
# Raise 3 to power of each number in arr
print(repr(np.power(3, arr)))
arr2 = np.array([[10.2, 4], [3, 5]])
# Raise arr2 to power of each number in arr
print(repr(np.power(arr2, arr)))

#output
array([[ 3,  9],
       [27, 81]])
array([[ 10.2,  16. ],
       [ 27. , 625. ]])
```
NumPy has various other mathematical functions, which are listed [here](https://numpy.org/doc/stable/reference/routines.math.html)

c. Matrix multiplication<br>
NumPy arrays are basically vectors and matrices, it makes sense that there are functions for dot products and matrix multiplication.The main function to use is np.matmul, which takes two vector/matrix arrays as input and produces a dot product or matrix multiplication.<br>
Nb: that the dimensions of the two input matrices must be valid for a matrix multiplication. Specifically, the second dimension of the first matrix must equal the first dimension of the second matrix, otherwise np.matmul will result in a ValueError.<br>
The code below shows various examples of matrix multiplication. When both inputs are 1-D, the output is the dot product.
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([-3, 0, 10])
print(np.matmul(arr1, arr2))

arr3 = np.array([[1, 2], [3, 4], [5, 6]])
arr4 = np.array([[-1, 0, 1], [3, 2, -4]])
print(repr(np.matmul(arr3, arr4)))
print(repr(np.matmul(arr4, arr3)))
# This will result in ValueError
print(repr(np.matmul(arr3, arr3)))

#output
27
array([[  5,   4,  -7],
       [  9,   8, -13],
       [ 13,  12, -19]])
array([[  4,   4],
       [-11, -10]])
Traceback
```
More illustrations on Math Numpy [here](https://github.com/veldakarimi/Machine_Learning/blob/debc22f21234830e610fbde1011088645ad973e0/numpymath.py#L1-L13)
<br>

### Random
Generate numbers and arrays from different random distributions.<br>
a. Random Integers <br>
Similar to Python's random module, NumPy has its own submodule for pseudo-random number generation called np.random.<br>
It provides all the necessary randomized operations and extends it to multi-dimensional arrays.<br>
To generate pseudo-random integers, we use the np.random.randint function.<br>
Example of np.random.randint:<br>
``` python
print(np.random.randint(5))
print(np.random.randint(5))
print(np.random.randint(5, high=6))

random_arr = np.random.randint(-3, high=14,
                               size=(2, 2))
print(repr(random_arr))

'''
output
4
2
5
array([[ 4,  2],
       [ 2, 12]])
'''
```
The np.random.randint function takes in a single argument, which depends on the high keyword argument.<br>
If high=None (default value), then the required argument represents the upper (exclusive) end of the range, with the lower end being 0. Specifically, if the required argument is n, then the random integer is chosen uniformly from the range [0, n).<br>

If high is not None, then the required argument will represent the lower (inclusive) end of the range, while high represents the upper (exclusive) end.<br>
The size keyword argument specifies the size of the output array, where each integer in the array is randomly drawn from the specified range. <br>
As a default, np.random.randint returns a single integer.<br>

b. Utility Functions<br>
Fundamental utility functions from the np.random module are np.random.seed and np.random.shuffle.<br>
We use the np.random.seed function to set the random seed, which allows us to control the outputs of the pseudo-random functions. <br>
The function takes in a single integer as an argument, representing the random seed.<br>
Random seed specifies the start point when a computer generates a random number sequence.<br>
Example using np.random.seed with the same random seed. <br>
Note how the outputs of the random functions in each subsequent run are identical when we set the same random seed.<br>
```python
np.random.seed(1)
print(np.random.randint(10))
random_arr = np.random.randint(3, high=100,
                               size=(2, 2))
print(repr(random_arr))

# New seed
np.random.seed(2)
print(np.random.randint(10))
random_arr = np.random.randint(3, high=100,
                               size=(2, 2))
print(repr(random_arr))

# Original seed
np.random.seed(1)
print(np.random.randint(10))
random_arr = np.random.randint(3, high=100,
                               size=(2, 2))
print(repr(random_arr))

'''
output
5
array([[15, 75],
       [12, 78]])
8
array([[18, 75],
       [25, 46]])
5
array([[15, 75],
       [12, 78]])
'''
```
np.random.shuffle function allows us to randomly shuffle an array. <br>
Note that the shuffling happens in place (i.e. no return value), and shuffling multi-dimensional arrays only shuffles the first dimension.<br>
Example of np.random.shuffle. Note that only the rows of matrix are shuffled (i.e. shuffling along first dimension only).<br>
```python
vec = np.array([1, 2, 3, 4, 5])
np.random.shuffle(vec)
print(repr(vec))
np.random.shuffle(vec)
print(repr(vec))

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
np.random.shuffle(matrix)
print(repr(matrix))

'''
output
array([5, 4, 3, 2, 1])
array([2, 5, 4, 3, 1])
array([[4, 5, 6],
       [7, 8, 9],
       [1, 2, 3]])
'''
```
C. Distributions <br>
Using np.random we can  draw samples from probability distributions.e.g, we can use np.random.uniform to draw pseudo-random real numbers from a uniform distribution.<br>
Example of np.random.uniform. <br>
```python 
print(np.random.uniform())
print(np.random.uniform(low=-1.5, high=2.2))
print(repr(np.random.uniform(size=3)))
print(repr(np.random.uniform(low=-3.4, high=5.9,
                             size=(2, 2))))

'''output
0.0012899920879422266
0.16014194484284805
array([0.42698393, 0.13810429, 0.0028385 ])
array([[-0.66067351,  5.27294206],
       [ 3.81933726, -2.22900708]])
'''
```
The function np.random.uniform actually has no required arguments. The keyword arguments, low and high, represent the inclusive lower end and exclusive upper end from which to draw random samples. Since they have default values of 0.0 and 1.0, respectively, the default outputs of np.random.uniform come from the range [0.0, 1.0). <br>

The size keyword argument  represents the output size of the array.<br>
Another distribution we can sample from is the normal (Gaussian) distribution. The function we use is np.random.normal.<br>
Example of np.random.normal:<br>
```python
print(np.random.normal())
print(np.random.normal(loc=1.5, scale=3.5))
print(repr(np.random.normal(loc=-2.4, scale=4.0,
                            size=(2, 2))))
                            
'''
output
0.6084710983104439
3.2985332151781908
array([[ 5.44064605,  2.92177381],
       [-6.99030241, -7.32588644]])
'''
```
Like np.random.uniform, np.random.normal has no required arguments.<br>
The loc and scale keyword arguments represent the mean and standard deviation, respectively, of the normal distribution we sample from.<br>
Other built in [distributions](https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.random.html)<br>

d. Custom Sampling <br>
While NumPy provides built-in distributions to sample from, we can also sample from a custom distribution with the np.random.choice function.<br>
```python
colors = ['red', 'blue', 'green']
print(np.random.choice(colors))
print(repr(np.random.choice(colors, size=2)))
print(repr(np.random.choice(colors, size=(2, 2),
                            p=[0.8, 0.19, 0.01])))

''' output
blue
array(['blue', 'green'], dtype='<U5')
array([['blue', 'red'],
       ['red', 'red']], dtype='<U5')
'''       
```

### Indexing 
Index into NumPy arrays to extract data and array slices.<br>
a. Array Accessing<br>
Accessing NumPy arrays is identical to accessing Python lists. For multi-dimensional arrays, it is equivalent to accessing Python lists of lists.<br>
Example:<br>
```python
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])
print(arr[4])

arr = np.array([[6, 3], [0, 2]])
# Subarray
print(repr(arr[0]))

'''output 
1
5
array([6, 3])
'''
```

b. Slicing<br>
NumPy arrays also support slicing.<br> Similar to Python, we use the colon operator (i.e. arr[:]) for slicing.<br>
We  also use negative indexing to slice in the backwards direction.

```python
arr = np.array([1, 2, 3, 4, 5])
print(repr(arr[:]))
print(repr(arr[1:]))
print(repr(arr[2:4]))
print(repr(arr[:-1]))
print(repr(arr[-2:]))

''' output
array([1, 2, 3, 4, 5])
array([2, 3, 4, 5])
array([3, 4])
array([1, 2, 3, 4])
array([4, 5])
```
For multi-dimensional arrays, we can use a comma to separate slices across each dimension.<br>
Example slices of a 2-D NumPy array:<br>
``` python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(repr(arr[:]))
print(repr(arr[1:]))
print(repr(arr[:, -1]))
print(repr(arr[:, 1:]))
print(repr(arr[0:1, 1:]))
print(repr(arr[0, 1:]))

'''
output
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
array([[4, 5, 6],
       [7, 8, 9]])
array([3, 6, 9])
array([[2, 3],
       [5, 6],
       [8, 9]])
array([[2, 3]])
array([2, 3])

'''
```

c. Argmin and argmax<br>
It is useful to figure out the actual indexes of the minimum and maximum elements.<br>
To do this, we use the np.argmin and np.argmax functions.<br>
Example usages of np.argmin and np.argmax. Note that the index of element -6 is index 5 in the flattened version of arr:<br>
``` python
arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [-3, 9, 1]])
print(np.argmin(arr[0]))
print(np.argmax(arr[2]))
print(np.argmin(arr))
''' output
2
1
5
'''
```
The np.argmin and np.argmax functions take the same arguments. The required argument is the input array and the axis keyword argument specifies which dimension to apply the operation on.<br>
Example of axis keyword argument is used for these functions:<br>
```python
arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [-3, 9, 1]])
print(repr(np.argmin(arr, axis=0)))
print(repr(np.argmin(arr, axis=1)))
print(repr(np.argmax(arr, axis=-1)))

'''output
array([2, 0, 1])
array([2, 2, 0])
array([1, 1, 1])
'''
```
[Indexing example](https://github.com/veldakarimi/Machine_Learning/blob/master/indexing.py#L1-L51)



















