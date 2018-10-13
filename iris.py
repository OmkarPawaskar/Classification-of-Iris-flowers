#Part 1 - Data Preprocessing
#Importing Libraries

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url,names = names)

# Summarize Dataset

#Dimensions of dataset
dataset.shape
#peek at data
dataset.head(20)#first 20 rows
#Statistical Summary
dataset.describe()
#Class Distribution
dataset.groupby('class').size()

# Part 2 - Data Visualization
#Univariate plots to better understand each attribute.
#Multivariate plots to better understand the relationships between attributes.

#Univariate plot
#box and whiskers plot
dataset.plot(kind = 'box',subplots = True,layout = (2,2), sharex = False ,sharey = False)
plt.show()

#histogram plot
dataset.hist()
plt.show()
#two of the input variables have a Gaussian distribution

#Multivariate plot
#to spot structured relationships between input variables.

#scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

# Part 3 - Evaluate Algorithms

# Create a Validation Dataset
#split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
from sklearn.model_selection import train_test_split
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size = validation_size,random_state = seed)

#Test Harness
#use 10-fold cross validation to estimate accuracy.
#This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
seed = 7
scoring = 'accuracy'

# Build Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

'''LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
CART: 0.975000 (0.038188)
NB: 0.975000 (0.053359)
SVM: 0.991667 (0.025000)'''

# Select Best Models

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#You can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.

# Make Predictions
# KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation set.

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions)) #0.9
print(confusion_matrix(Y_validation, predictions))
'''[[ 7  0  0]
 [ 0 11  1]
 [ 0  2  9]]'''
print(classification_report(Y_validation, predictions))


