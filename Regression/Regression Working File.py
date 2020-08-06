import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("Regression/student-mat.csv", sep=";")  # normally comma's
# print(data.head()) # Compare this data with below data

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"  # Final Grade

x = np.array(data.drop([predict], 1))  # features/attributes
y = np.array(data[predict])  # labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

'''
# Finds best ML model
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Split data so the model hasn't seen everything, spliting it 10%

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test) # Calculates accuracy
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:  # Saving model
            pickle.dump(linear, f)  # Saves pickle file

'''

# Above is commented out to prove pickle file saved our ML model
pickle_in = open("Regression/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Coefficient: \n", linear.coef_)  # slopes of line per each dimension, higher = more influence
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])  # predictions, input data, actual output data

p = "G1" # Use this to see what factors correlate by changing attributes
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
