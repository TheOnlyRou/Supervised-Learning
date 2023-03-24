# Classification
# Project: Predicting whether a credit card application will be approved or not.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("\nClassification KNN model accuracy for the IRIS dataset")
# https://archive.ics.uci.edu/ml/datasets/iris
# Iris is a famous classification dataset consists of 3 types of irises (Setosa, Versicolor, Virginica)
# Samples per class: 50 so total samples: 150
# Sepal Length | Sepal Width | Petal Length | Petal Width
# data: features
# target: name of class
# Load the iris dataset
iris = load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=98)
# test_size: proportion of test dataset to include in test (0 < x < 1)
# random_state: integer that controls shuffling in dataset. Use an integer you know for reproducible output

# Train a KNN classifier on the training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the classifier on the test set
accuracy = knn.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Regression:
# Project: Predicting the price of a house based on its features.
# Number of dataset instances: 20640, 8 attributes with no missing attribute values
# target is median house value expressed in hundreds of Thousands
# https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
# Load the California Housing dataset
#
housing = fetch_california_housing()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.7, random_state=42)
# test_size: proportion of test dataset to include in test (0 < x < 1)
# random_state: integer that controls shuffling in dataset. Use an integer you know for reproducible output

# Train a linear regression model on the training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Regression Model MSE & R2 Score for the California Housing Dataset")
print("Mean squared error: {:.2f}".format(mse))
print("R2 score: {:.2f}".format(r2))
