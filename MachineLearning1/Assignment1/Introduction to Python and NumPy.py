#!/usr/bin/env python
# coding: utf-8

# # Introduction to Python and Numpy/Scipy

# ###### What is Python?
# 
# [Python](http://www.python.org/) is a modern, general-purpose, object-oriented, high-level programming language.
# 
# From [the python website](https://www.python.org/):
# 
# > Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python’s elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms.
# 
# We’ll be using **Python 3 (v3.7+ recommended)!**

# General characteristics of Python:
# 
# * **clean and simple language:** Easy-to-read and intuitive code, easy-to-learn minimalistic syntax, maintainability scales well with size of projects.
# * **expressive language:** Fewer lines of code, fewer bugs, easier to maintain.

# Technical details:
# 
# * **dynamically typed:** No need to define the type of variables, function arguments or return types.
# * **automatic memory management:** No need to explicitly allocate and deallocate memory for variables and data arrays. No memory leak bugs. 
# * **interpreted:** No need to compile the code. The Python interpreter reads and executes the python code directly.

# **Advantages:**
# 
# * The main advantage is ease of programming, minimizing the time required to develop, debug and maintain the code.
# * Well designed language that encourage many good programming practices.
# * A large standard library, and a large collection of add-on packages.

# **Disadvantages:**
# 
# * Since Python is an interpreted and dynamically typed programming language, the execution of python code can be slow compared to compiled statically typed programming languages, such as C and Fortran. 
# * Somewhat decentralized, with different environment, packages and documentation spread out at different places. Can make it harder to get started.

# ## Why Python?

# * **Open-source language** that works everywhere (including the Raspberry Pi!)
# 
# * **Extensive ecosystem** of scientific libraries and environments
# 
# * **Great performance** due to close integration with time-tested and highly optimized codes written in C and Fortran:
#     * blas, altas blas, lapack, arpack, Intel MKL, ...

# * Good support for 
#     * **Parallel processing** with processes and threads
#     * **Interprocess communication** (MPI)
#     * **GPU** computing (OpenCL and CUDA)
#         * Eg. [Tensorflow](https://pytorch.org/), [PyTorch](https://pytorch.org/), [Numba](https://numba.pydata.org/), ....

# * Python has a strong position in scientific computing: 
#     * Large community of users, easy to find help and documentation.
# * Readily available and suitable for use on high-performance computing clusters.

# ### The scientific python software stack -- SciPy stack
# 
# * **NumPy** defines the numerical array and matrix types and basic operations on them 
# <img src="images/numpy.png" width="200" style="float:right">
# * The **SciPy library**, a collection of numerical algorithms and domain-specific toolboxes, including signal processing, optimization, statistics and much more.
# * **Matplotlib** for plotting
# * **Ipython** for running python code interactively
# 
# 
# 
# 

# ## Installation
# 
# ### Cross platform
# * (**RECOMMENDED**) For **Linux/Mac/Windows** use 
#     * Anaconda - https://www.anaconda.com/distribution/
#     * Miniconda - https://docs.conda.io/en/latest/miniconda.html

# ## IDEs

# ### PyCharm (Recommended)
# * Download at https://www.jetbrains.com/pycharm/
# * General purpose Python IDE with good NumPy, SciPy, IPython support
# * Can get professional version using your academic email address
#   * https://www.jetbrains.com/shop/eform/students

# ### Spyder 
# * Download at https://www.spyder-ide.org
# * Specifically for Scientific python
# * MATLAB-like interface if you're familiar with MATLAB

# ### Other options
# * Atom
# * Visual Studio Code
# * vim with appropriate plugins
# * emacs

# ### IPython 
# 
# * Interactive shell to run python code
# * Different from the normal python shell
# * Really good for trying out small snippets of code

# # Introduction to Python

# ## Python program files
# 
# * Python code is usually stored in text files with the file ending "`.py`":
# 
#         myprogram.py
#         
# * To run our Python program from the command line we use:
# 
#         $ python myprogram.py
#         
# 

# * Every line in a Python program file is assumed to be a Python statement, or part thereof. 
# 
#     * The only exception is comment lines, which start with the character `#` (optionally preceded by an arbitrary number of white-space characters, i.e., tabs or spaces). Comment lines are usually ignored by the Python interpreter.

# In[ ]:


# this is the first comment
spam = 1  # and this is the second comment
          # ... and now a third!
text = "# This is not a comment because it's inside quotes."


# ### Comments, assignment, variables
# 
#   * Single-line comments start wtih a `#`, multiline comments are enclosed in `"""` (three double quotes)
#   * Assignment is done with `=`

# In[ ]:


a = 1
a


#   * Variable names in Python can contain alphanumerical characters `a-z`, `A-Z`, `0-9` and some special characters such as `_`. Normal variable names must start with a letter. 
#     * Conventions: Classes are in `CamelCase`, local variables in `under_score_case`, constants in `UPPER_CASE_CHARACTERS`

# ### Strings, print
# 
#   * A variable has to be defined before it can be used
#   * Strings are enclosed in either double-quotes (`"`) or single quotes (`'`)
#   * Use the `print` function to print output in scripts. (In `ipython` it automatically prints the output)

# In[ ]:


# Error!
print(st)


# In[ ]:


s = "Hello, World!"
print(s)


# ### Arithmetic and logical operators
# 
#   * Usual **arithmetic** operators exist: 
#     * `+`, `-`, `/`, `*` for addition, subtraction, division and multiplication respectively
#     * `%` used for remainder of division, `**` for calculating powers
#   * **Comparison** is done using: `<`, `<=`, `>`, `>=`, `!=`, `is`, `in`
#   * **Logical** operators are: `and`, `or`, `not`

# In[ ]:


(50 - 5 * 6) // (3 + 1)


# In[ ]:


17 % 3


# In[ ]:


5 ** 2


# In[ ]:


5 > 20


# In[ ]:


a = None
a is None


# In[ ]:


a = 10
a is not None


# In[ ]:


b = 12
a > 2 and b < 2


# In[ ]:


not a > 2 


# ### Types
# 
# * The basic types in Python are `int`, `float`, `boolean`, `complex numbers`, `bytes`
# * The `type` function gives you the type of a variable

# * **NOTE:** Python is ***dynamically*** typed but ***strongly*** typed! [[1]](https://wiki.python.org/moin/Why%20is%20Python%20a%20dynamic%20language%20and%20also%20a%20strongly%20typed%20language)

# In[ ]:


i = 10  # integer
f = 10e6  # float
b = False  # Boolean is True or False
c = 2 + 3j  # Complex
print("The types of the variables are:\ni\t{}\nf\t{}\nb\t{}\nc\t{}\n".format(type(i), type(f), type(b),  type(c)))


# In[ ]:


s = "Hello "  # string
i = 10  # Integer
print(s + i)  # Python doesn't know how to add an integer and string!


# In[ ]:


print(s + str(i))  # You have to explicitly cast the variable


# ### List
# 
# *Python knows a number of compound data types, used to group together other values. The most versatile is the list, which can be written as a list of comma-separated values (items) between square brackets. Lists might contain items of different types, but usually the items all have the same type.*
# 
# * It allows slicing, indexing, etc.
# 
# **Note for MATLAB users:** Indexing start at 0!

# In[ ]:


squares = [1, 4, 9, 16, 25]
squares


# In[ ]:


len(squares)


# In[ ]:


squares[0]


# In[ ]:


squares[2:4]  # Slice -- note the inclusive / exclusive behaviour!


# In[ ]:


squares.append(36)
squares


# ### Dictionary
# Dictionaries are also like lists, except that each element is a key-value pair. The syntax for dictionaries is `{key1 : value1, ...}`:

# In[ ]:


dictionary = {"one": 1, "two": 2}
dictionary


# In[ ]:


another_dictionary = {3: "three", 4: "four"}
another_dictionary


# In[ ]:


yet_another_dictionary = {"Five": "five", "six": 6.0, "seven": 7, 8: 'eight'}
yet_another_dictionary


# ### Indentation!
# * In Python, spaces at the beginning of a line i.e. indentation is significant!!
# * There is no equivalent of the curly braces in 'C'. 
#     * My convention is to always use spaces (because different systems handle tabs in different ways)
#     * Most IDEs can be set-up to automatically convert a tab into 2/4 spaces

# ### Control flow, functions
#   * Control flow using: `if`, `while`, `for`
#   * Define functions using: `def`

# In[ ]:


# if
if 21 < 42:
    print("This is only half the truth!")
    print("asdf")


# In[ ]:


# for loop
numbers = [1, 2, 3, 4]
cubes = []
for number in numbers:          # Notice the colon at the end of the line!
    cubed_number = number ** 3  # Notice the indentation!!
    cubes.append(cubed_number)  # Indentation again!!
cubes


# In[ ]:


def answer():
    return 42

def is_answer(x):
    return x == answer()

is_answer(42)


# ### import
# 
#   * Use python libraries using the `import` statement
#   * Python has a vast, very functional [standard library](https://docs.python.org/3/tutorial/stdlib.html)

# In[ ]:


import math  # From the standard library

def sqrt(number):  # colon again!
    return math.sqrt(number)

sqrt(256)


# In[ ]:


from math import pi  # import specific function / module
from math import cos as cosine  # apply alias
cosine(pi)


# ## Further reading for Python
# 
#  * [Python](http://www.python.org). The official Python web site.
#  * [Python tutorials](http://docs.python.org/3/tutorial). The official Python tutorials.
#  * [Free Dive into Python book](http://diveinto.org/python3/)
#  * [Learn X in Y minutes](https://learnxinyminutes.com/docs/python3/)
#  * I highly recommend you follow the style conventions described on the Python website in [PEP0008](https://www.python.org/dev/peps/pep-0008/)
# 

# # NumPy

# * The `numpy` package (module) is used in almost all numerical computation using Python.
# * It is a package that provide high-performance vector, matrix and higher-dimensional data structures for Python.
# * It is implemented in C and Fortran so when calculations are vectorized (formulated with vectors and matrices), performance is very good. 
# 
# To use `numpy` you need to import the module, using:

# In[ ]:


import numpy as np


# ### Creating `numpy` arrays
# 
# In the `numpy` package the terminology used for vectors, matrices and higher-dimensional data sets is *array*. 
# 
# There are a number of ways to initialize new numpy arrays, for example from
# 
# * a Python list or tuples
# * using functions that are dedicated to generating numpy arrays, such as `arange`, `linspace`, etc.
# * reading data from files

# In[ ]:


v = np.array([1,2,3,4])  # From a list
v


# In[ ]:


M = np.array([[1, 2], [3, 4]])  # From a nested list i.e. a 2-dimensional list
M  # Two-dimensional matrix


# In[ ]:


x = np.arange(1, 20, 2)  # Arguments are beginning (1), end (20), step (2)
x  # All odd numbers


# In[ ]:


y = np.linspace(1, 20, 10)  # Arguments are beginning  (1), end (20), number of numbers to generate in between (10)
y


# ### Other array generation methods
#   * Generate random data using: `np.random`
#   * Generate arrays filled with 0s or 1s using: `np.zeros` and `np.ones`

# In[ ]:


from numpy.random import default_rng
rng = default_rng()
R = rng.random((3, 3))  # 3x3 matrix of random numbers between 0 and 1, uniform distribution
R


# In[ ]:


rng = default_rng(42)  # notice the seed of the RNG
R = rng.random((3, 3))  # 3x3 matrix of random numbers between 0 and 1, uniform distribution
R


# In[ ]:


np.zeros((3, 3))  # 3x3 matrix of zeroes


# In[ ]:


np.ones((3, 3))  # 3x3 matrix of ones


# ### Array properties
#   * Use `.shape` to get dimensions
#   * Use `.dtype` to get type of contents of matrix

# In[ ]:


x.shape


# In[ ]:


x.dtype


# In[ ]:


x.ndim


# ### Indexing
#   * Similar to C indexing for 1-dimensional case -- `[.]`
#   * Use `[.,.]` for 2-D indexing
#   * Use `:` for slicing/range

# In[ ]:


R = rng.random((10, 3))  # matrix of random numbers between 0 and 1
R


# In[ ]:


R[1, 2]  # Element in row with index 1 and column with index 2 i.e. second row, third column


# In[ ]:


R[:, 1]  # Select all rows at the column with index 1 i.e. the entire 2nd column (Remeber, 0 indexing)


# In[ ]:


R[2, :]  # Select the row with index 2 and all columns i.e. the entire 3rd row


# In[ ]:


R[0:6:2, :]


# In[ ]:


R[np.arange(0, 6, 2), :]


# In[ ]:


rand_bool = R > 0.5
rand_bool


# In[ ]:


R[rand_bool]


# ### Arithmetic operations
#   * All normal arithmetic operators (`+`,`-`,`*`,`/` etc.) perform **elementwise** operations! (***NOTE!***)

# In[ ]:


A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])


# In[ ]:


A


# In[ ]:


B


# In[ ]:


A + B

A * B  # This is element-wise multiplication, NOT the dot product

A / B  # Again, element-wise division


# ### Dot product
#   * Use `.dot` method of the array
#   * Or use the `np.dot` function
#   * Or use the `@` shorthand for `np.matmul` (see [the documentation](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) for differences between `np.dot` and `np.matmul`)

# In[ ]:


print(A.shape)
print(B.shape)
A.dot(B)


# In[ ]:


# Which is the same as:
np.dot(A, B)


# In[ ]:


# Which is the same as:
A @ B


# In[ ]:


# Vector matric multiplication
v1 = np.arange(0, 4)
A = rng.random((4, 4)) * 10  # 2-D == matrix. This is a float64 array
A = A.astype(np.int)  # Force the array type to int


# In[ ]:


v1


# In[ ]:


A


# In[ ]:


# Dot product
np.dot(v1, A)


# In[ ]:


# Is different from
np.dot(A, v1)


# * Transpose using `np.transpose` or `.T`
# * Inverse using `np.linalg.inv`

# In[ ]:


A


# In[ ]:


A.T


# In[ ]:


np.linalg.inv(A)


# In[ ]:


np.linalg.pinv(A)


# ## Data processing
#   * Utility functions: `np.mean`, `np.std`, `np.min`, `np.max`, `np.sum`

# In[ ]:


a = np.arange(10)
a


# In[ ]:


print("Avg: ", np.mean(a))

print("Std: ", np.std(a))

print("Min: ", np.min(a))

print("Max: ", np.max(a))

print("Sum: ", np.sum(a))


# ## Further reading
# * https://numpy.org/ - NumPy homepage
# * https://numpy.org/doc/stable/user/quickstart.html - NumPy quick start guide
# * https://numpy.org/doc/stable/user/numpy-for-matlab-users.html - A NumPy guide for MATLAB users.

# ---
# # Plotting
# * Use [matplotlib](http://matplotlib.org)'s `pyplot`
# * Usage Guide -- https://matplotlib.org/stable/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
# * Pyplot Tutorial -- https://matplotlib.org/stable/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


# #### Formatting your plots

# In[ ]:


plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'go--')


# In[ ]:


# Alternative
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], marker='o', c='r')


# Issuing multiple `pyplot` calls (`plot`, `scatter`, ...) one after another produces a single figure with multiple "layers". You can show the result by issuing `plt.show()` in Python scripts or execute the Jupyter notebook cell with `%matplotlib inline` enabled.

# In[ ]:


# evenly sampled points
t = np.arange(0., 5., 0.2)
# Compose the figure
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'bs')
plt.plot(t, t**3, 'g^')


# #### Subplot

# In[ ]:


names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')  # Set the plot title


# In[ ]:


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


# # scikit-learn
# 
# * Website -- https://sklearn.org/
# * Tutorials -- https://sklearn.org/tutorial/basic/tutorial.html
# * User Guide -- https://sklearn.org/user_guide.html

# ## Intro to ML with scikit-learn
# scikit-learn comes with a few standard datasets, for instance the `iris` and `digits` datasets for classification and the `boston house prices` dataset for regression.

# In[6]:


from sklearn import datasets
digits = datasets.load_digits()


# A dataset is a dictionary-like object that holds all the data and some metadata about the data. This data is stored in the `.data` member, which is a `n_samples, n_features` array. In the case of supervised problem, one or more response variables are stored in the `.target` member. More details on the different datasets can be found in the documentation.

# In[7]:


digits.data.shape


# In[8]:


digits.data[0]


# In[9]:


print(digits.target[0])


# In[10]:


fig, axss = plt.subplots(5, 5, figsize=(10, 10))
axs = axss.flatten()
for i in range(25):
    ax = axs[i]
    ax.imshow(digits.data[i].reshape(8, 8))


# In the case of the digits dataset, the task is to predict, given an image, which digit it represents. We are given samples of each of the 10 possible classes (the digits zero through nine) on which we fit an estimator to be able to predict the classes to which unseen samples belong.
# 
# In scikit-learn, an estimator for classification is a Python object that implements the methods `fit(X, y)` and `predict(T)`.
# 
# An example of an estimator is the class `sklearn.svm.SVC` that implements support vector classification. The constructor of an estimator takes as arguments the parameters of the model, but for the time being, we will consider the estimator as a black box.

# In[15]:


from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)


# We call our estimator instance `clf`, as it is a classifier. It now must be fitted to the model, that is, it must learn from the model. This is done by passing our training set to the `fit` method. As a training set, let us use all the images of our dataset apart from the last one. We select this training set with the `[:-1]` Python syntax, which produces a new array that contains all but the last entry of `digits.data`:

# In[16]:


clf.fit(digits.data[:-1], digits.target[:-1])


# Now you can predict new values, in particular, we can ask to the classifier what is the digit of our last image in the `digits` dataset, which we have not used to train the classifier:

# In[17]:


print("Prediction: ", clf.predict(digits.data[-1:]))
print("True label: ", digits.target[-1:])

fig, ax = plt.subplots()
ax.imshow(digits.data[-1:].reshape(8, 8))


# # Attributions and License:
# 
# Notebook based on [http://github.com/jrjohansson/scientific-python-lectures](http://github.com/jrjohansson/scientific-python-lectures) by
# J.R. Johansson (jrjohansson at gmail.com).
# 
# Some code snippets borrowed from [The Python Tutorial](https://docs.python.org/3.5/tutorial/index.html)
# 
# The scikit-learn code is borrowed from [The scikit-learn documentation](https://sklearn.org/tutorial/basic/tutorial.html)
# 
# This work is licensed under a [Creative Commons Attribution 3.0 Unported License.](http://creativecommons.org/licenses/by/3.0/)

# In[ ]:




