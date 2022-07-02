# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
# Import Libraries
import pandas as pd
import numpy as np
import os
import cv2
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import img_to_array, load_img
from glob import glob

from sklearn.metrics import confusion_matrix, classification_report

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

path = "C:/Users/naufa/OneDrive - Universiti Kebangsaan Malaysia/Desktop/shrdc ai technologist/deep learning/module/Concrete Crack Images for Classification"

pos_images = glob(f"C:/Users/naufa/OneDrive - Universiti Kebangsaan Malaysia/Desktop/shrdc ai technologist/deep learning/module/Concrete Crack Images for Classification/Positive/*.jpg");
pos_labels = [1] * len(pos_images)
neg_images =  glob(f"C:/Users/naufa/OneDrive - Universiti Kebangsaan Malaysia/Desktop/shrdc ai technologist/deep learning/module/Concrete Crack Images for Classification/Negative/*.jpg");
neg_labels = [0] * len(neg_images)
#%%

pos_image_list = []
for i in pos_images:
    image = cv2.imread(i)
    image_resize = cv2.resize(image, (100, 100))
    pos_image_list.append(image_resize)

pos_image_resized = np.array(pos_image_list)

neg_image_list = []
for i in neg_images:
    image = cv2.imread(i)
    image_resize = cv2.resize(image,(100,100))
    neg_image_list.append(image_resize)
    
neg_image_resized = np.array(neg_image_list)
    

#%%

np.save("positive_resize",pos_image_resized)
np.save("negative_resize",neg_image_resized)


#%%

pos_resized = np.load("positive_resize.npy")
neg_resized = np.load("negative_resize.npy")


#%%

final_train = []
for i in pos_resized[0:14000]:
    final_train.append([i,1])
    
for j in neg_resized[0:14000]:
    final_train.append([j,0])
# pos_train = pos_resized[0:15000]
# pos_label_train = [1]*len(pos_train)

# neg_train = neg_resized[0:15000]
# neg_label_train = [0]*len(neg_train)

#%%

len(final_train)

#%%

final_train = np.array(final_train,dtype=object)
final_train = shuffle(final_train)
images_train = []
labels_train = []

for i in final_train:
    images_train.append(i[0])
    labels_train.append(i[1])

#%%
# final_valid = []
# for i in pos_resized[15000:19000]:
#     final_valid.append([i,1])
    
# for j in neg_resized[15000:19000]:
#     final_valid.append([j,0])



final_test = []

for i in pos_resized[14000:20000]:
    final_test.append([i,1])
    
for j in neg_resized[14000:20000]:
    final_test.append([j,0])
    
final_test = np.array(final_test,dtype=object)
final_test = shuffle(final_test)
images_test = []
labels_test = []

for i in final_test:
    images_test.append(i[0])
    labels_test.append(i[1])

# pos_valid_test = pos_resized[15000:19000]
# pos_label_valid = [1]*len(pos_valid_test)
# neg_valid_test = neg_resized[15000:19000]
# neg_label_valid = [0]*len(neg_valid_test)

#%%

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(100,100,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#%%

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

#%%

print(model.summary())

#%%

print(len(images_train),len(labels_train))
print(len(images_test),len(labels_test))

images_train = np.array(images_train)
labels_train = np.array(labels_train)


images_test = np.array(images_test)
labels_test = np.array(labels_test)

#%%

model_history = model.fit(images_train, labels_train, validation_data = (images_test, labels_test), epochs = 10, batch_size=128)

#%%
# final_test = np.concatenate((pos_test,neg_test),axis=0)
# final_label_test = np.concatenate((pos_label_test,neg_label_test),axis=0)

prediction = model.predict(images_test)
k=(prediction >= 0.5).astype(np.int)
print("Accuracy_Score : {}".format(accuracy_score(k, labels_test) * 100))

#%%

prediction

#%%

k=(prediction >= 0.5).astype(np.int)

#%%
# final_test = np.concatenate((pos_test,neg_test),axis=0)
# final_label_test = np.concatenate((pos_label_test,neg_label_test),axis=0)

prediction = model.predict(images_test)
print("Accuracy_Score : {}".format(accuracy_score(k, labels_test) * 100))

#%%


fig = px.line(
    model_history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()


fig = px.line(
    model_history.history,
    y=['accuracy', 'val_accuracy'],
    labels={'index': "Epoch", 'value': "Accuracy"},
    title="Training and Validation Accuracy Over Time"
)

fig.show()
#%%
results = model.evaluate(images_test, labels_test, batch_size=128)
print("test loss, test acc:", results)

#%%

def evaluate_model(model, images_test,labels_test):
    
    results = model.evaluate(images_test,labels_test, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
    y_pred = np.squeeze((model.predict(images_test) >= 0.5).astype(np.int))
    cm = confusion_matrix(labels_test, y_pred)
    clr = classification_report(labels_test, y_pred, target_names=["NEGATIVE", "POSITIVE"])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr)
#%%
evaluate_model(model,images_test,labels_test )