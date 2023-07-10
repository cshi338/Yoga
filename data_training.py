import os  
import json
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

 
is_init = False
size = -1

label = []
dictionary = {}
c = 0
yoga_poses = {}
for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
			yoga_poses[i.split(".")[0]] = size
		else:
			X = np.concatenate((X, np.load(i)))
			size = np.load(i).shape[0]
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))
			print(X.shape, y.shape, size)
			yoga_poses[i.split(".")[0]] = size
		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1
print(yoga_poses)

for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")


y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))

m = Dense(128, activation="tanh")(ip)
m = Dense(64, activation="tanh")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.summary()

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

history = model.fit(X_new, y_new, epochs=300,
    validation_split=0.1)

#with open('history.json', 'w') as f:
    #json.dump(history.history, f)

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()

# Accuracy curve
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training accuracy', 'Validation accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
#plt.show()    

model.save("model.h5")
np.save("labels.npy", np.array(label))
