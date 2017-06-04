import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

from PIL import Image
import glob
import csv


import keras as k
from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten

import cv2
from tqdm import tqdm

x_train = []
x_train_add = []
x_test = []
x_test_add = []
y_train = []
y_test = []
y_train_add = []

x_names = []
#################################################################################################################################
#GETS DATA
df_train = pd.read_csv('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/train_v2.csv')
df_test = pd.read_csv('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/sample_submission_v2.csv')
df_test_add = df_test.tail(20522)
df_test = df_test.head(40669)

x_names = df_train['image_name'].values.tolist()
x_names = [s + 'gen' for s in x_names]

print(df_test_add)
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
print(label_map, inv_label_map)
for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (64, 64)))
    y_train.append(targets)

    
'''
filenames = []
for filename in glob.glob('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/train-generated/*.jpg'):
    img = cv2.imread(filename)
    filenames.extend([filename])

for value in x_names:
  for filename in filenames:
    if value in filename:
      file = filename
      print(file)
      img = cv2.imread(filename)
      x_train_add.append(cv2.resize(img, (48, 48)))
      '''
########################################################################################################################################

#with open('Y_TRAINGEN.csv', 'r') as f:
#    reader = csv.reader(f)
#    y_train_add = list(reader)
 
#y_train_add = pd.DataFrame(data=y_train_add, columns=labels)

#for label in labels:
#  y_train_add[label] = y_train_add[label].map({'1.0': 1, '0.0': 0})

#y_train_add = y_train_add.dropna()
#y_train_add = y_train_add.values.tolist()

#y_train_add = y_train.copy()

for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/test-jpg/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (64, 64)))

for f, tags in tqdm(df_test_add.values, miniters=1000):
    img = cv2.imread('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/test-jpg-additional/{}.jpg'.format(f))
    x_test_add.append(cv2.resize(img, (64, 64)))

#y_train = y_train + y_train_add
y_train = np.array(y_train, np.uint8)

#x_train = x_train + x_train_add
x_train = np.array(x_train, np.float16) / 255.

#print(len(x_train),len( x_train_add), len(y_train), len(y_train_add))


split = 35000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]


learn_rate=0.001; epoch=5; batch_size=128; validation_split_size=0.2; train_callbacks=(); output_size=17;

img_size=(64, 64)
img_channels=3

classifier = Sequential()

classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(256, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.25))


classifier.add(Flatten())

classifier.add(Dense(512, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(output_size, activation='sigmoid'))


X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=validation_split_size)

opt = keras.optimizers.Adam(lr=0.001)

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# early stopping will auto-stop training process if model stops learning after 3 epochs
earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

classifier.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[ *train_callbacks, earlyStopping])
fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
'''model.fit(x_train, y_train,
          batch_size=128,
          epochs=4,
          verbose=1,
          validation_data=(x_valid, y_valid))'''


datagen = ImageDataGenerator(
  rotation_range=10,
  width_shift_range=0.2,
  height_shift_range=0.2,
  zoom_range=0.1,
  horizontal_flip=True)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train)/32,
          epochs=30,
          verbose=1,
          validation_data=(x_valid, y_valid))

model.fit(x_train, y_train, batch_size=64,
          epochs=30,
          verbose=1,
          validation_data=(x_valid, y_valid))

x_test = np.array(x_test, np.float16) / 255.
x_test_add = np.array(x_test_add, np.float16) / 255.


def probToLabel(y, labels=labels):
  for label in labels:
    y.loc[y[label] == True, label] = label
  for label in labels:
    y.loc[y[label] == False, label] = ''

def labelsToTags(y, end=labels[-1]):
  tags = []
  all_tags = ''
  for count in range(len(y)):
    print(count)
    all_tags = ''
    for label in labels:
      all_tags = all_tags + y[label][count] + ' '
      if label == end:
        tags.extend([all_tags])
  return tags

#prints the fscore
def fscore(x_valid, y_valid, epsil=0.15483):
  p_valid = model.predict(x_valid, batch_size=128)
  print(fbeta_score(y_valid, np.array(p_valid) > epsil, beta=2, average='samples'))

def open_ytrain_add:
  with open('Y_TRAINGEN.csv', 'r') as f:
    reader = csv.reader(f)
    y_train_add = list(reader)
  y_train_add = pd.DataFrame(data=y_train_add, columns=labels)

  for label in labels:
    y_train_add[label] = y_train_add[label].map({'1.0': 1, '0.0': 0})

  y_train_add = y_train_add.dropna()
  y_train_add = y_train_add.values.tolist()

  def open_xtrain_add:
    for filename in glob.glob('/home/pikachu/MLProjects/AmazonUnderstandingPlanet/train-generated/*.jpg'):
    img = cv2.imread(filename)
    x_train_add.append(cv2.resize(img, (48, 48)))


y = pd.DataFrame(model.predict(x_test, batch_size=128) > .5)
y = y.rename(index=str, columns=inv_label_map)

probToLabel(y)
tags = labelsToTags(y)

y_test_labels_DFTEST = pd.DataFrame({'image_name': df_test['image_name'],'tags':tags})

y = pd.DataFrame(model.predict(x_test_add, batch_size=128) > .5)
y = y.rename(index=str, columns=inv_label_map)

probToLabel(y)
tags = labelsToTags(y)

y_test_labels_DFTESTADD = pd.DataFrame({'image_name': df_test_add['image_name'],'tags':tags})
y_test_labels = pd.concat([y_test_labels_DFTEST, y_test_labels_DFTESTADD], ignore_index=True)
y_test_labels.to_csv('Y_PRED1.csv', index=False)


from sklearn.metrics import fbeta_score
fscore(x_valid, y_valid)


print('accuracy', ((model.predict(x_valid) > .16) == y_valid).mean())