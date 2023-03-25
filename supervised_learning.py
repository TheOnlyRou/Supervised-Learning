# Classification
# Project: Predicting whether a credit card application will be approved or not.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_diabetes
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
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=3)
# test_size: proportion of test dataset to include in test (0 < x < 1)
# random_state: integer that controls shuffling in dataset. Use an integer you know for reproducible output

# Train a KNN classifier on the training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the classifier on the test set
accuracy = knn.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Regression:
# Project: Providing a quantitative measure of disease progression one year after baseline.
# Number of dataset instances: 442, 10 attributes with no missing attribute values
# target is quantitative measure of disease progression one year after baseline
# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
# Load the diabetes dataset
#
diabetes = load_diabetes()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.7, random_state=42)
# test_size: proportion of test dataset to include in test (0 < x < 1)
# random_state: integer that controls shuffling in dataset. Use an integer you know for reproducible output

# Train a linear regression model on the training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Regression Model MSE & R2 Score for the Diabetes Dataset")
print("Mean squared error: {:.2f}".format(mse))
print("R2 score: {:.2f}".format(r2))
