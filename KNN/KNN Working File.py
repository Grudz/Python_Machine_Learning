import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")  # Pandas data frame
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))  # vhigh = 3, others are lower
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# print(buying)

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))  # features (zip creates tuples)
y = list(cls)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)  # 0.2 = more test data for model but less to train on
# print(x_train, y_test)

model = KNeighborsClassifier(n_neighbors=9)  # Amount of neighbors

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]  # convert data back to text

'''
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)  # Come in as 2D but we are avoiding it
    print("N: ", n)  # Prints out distance between nieghbors
'''

