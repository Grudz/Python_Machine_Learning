# refer to: https://www.tensorflow.org/tutorials/keras/classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import keras
from tensorflow import keras  # Old tensorflow, that's why I am imported keras seperatly

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()  # keras makes it easy to load data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print(train_labels[0]) # These labels are defined on tensor flow website


train_images = train_images/255.0  # Want values between -1 and 1, between 0 and 1 here
#test_labels = test_labels/255.0
'''
print(train_images[7])  # Shows actual image value

plt.imshow(train_images[7], cmap=plt.cm.binary) # Cmap makes it look grayscale like it should be
plt.show()
'''

# All we have to do to create a NN
model = keras.Sequential([  # Sequential means create layers in order
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer flattened 28*28=784
    keras.layers.Dense(128, activation="relu"),  # Dense layer = fully connected layer, with rectify linear unit activation function - fast and versitle
    keras.layers.Dense(10, activation="softmax")  # Output layer, softmax = picks values so ouput layers add up to 1 (like probabilty)
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # optimize = atom (typical), loss = (popular), metrics=helps make loss lower

model.fit(train_images, train_labels, epochs=5)  # epochs - how many times you will see same image. (order of images affects network), have to play with this

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc", test_acc)  # 10$ accuracy?

predictions = model.predict(test_images)  # Shows 10 seperate probabilities
#prediction = model.predict(np.array([test_images[7]])) # Specific image test
#print(prediction[0]) # Probabilities output

plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predicted: " + class_names[np.argmax(predictions[i])])
    plt.show()

print(class_names[np.argmax(predictions[0])])  # Takes largest prediction in list and outputs it (This is what NN thinks is right)

